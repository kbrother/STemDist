import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.mtgnn import gtnet
from model.node_embed import NodeEmbedding_rnn, NodeEmbedding_birnn, NodeEmbedding_attn
import torch.optim as optim
import math
import sys
import copy
import torch.nn.functional as F


def test_syn(args, data, raw_data, device):
    synx = raw_data['x'].to(device)
    syny = raw_data['y'].to(device)
    print("load finish")
    print(synx.shape)
    print(syny.shape)

    scaler = dataloader['scaler']
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]    
    seq_len = dataloader['train_loader'].xs.shape[1]    
    out_dim = 1
    _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim)      
    _model.to(device)
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=args.lr)
    _model = _model.to(device)
   
    nm_input_real = np.mean(dataloader['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
    nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
    nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
    nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim
    
    nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
    nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
    nm_input_syn = torch.reshape(nm_input_syn, (synx.shape[2], -1))
    min_val_mse = sys.float_info.max

    min_val_loss = sys.float_info.max
    for i in tqdm(range(args.epochs)):
        _model.train()
        _model.embed_forward(nm_input_syn)
        output_syn = _model(synx.transpose(1, 3)).squeeze()
        loss_syn = F.mse_loss(output_syn, syny)
            #loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%args.check == 0:
            with torch.no_grad():
                _model.embed_forward(nm_input_real)
                if args.ae:
                    val_loss = _model.test_model(data['val_loader'], scaler, device, args.ae)
                else:
                    val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
    
            print(f"epoch :{i}, train loss: {loss_syn}, val loss: {val_loss}")
            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():    
        _model.embed_forward(nm_input_real)
        if args.ae:
            test_loss = _model.test_model(data['test_loader'], scaler, device, args.ae)
        else:
            test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))
    print(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")      
    with open(args.save_path, "a") as f:
        f.write(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")


# python -m model.load_mtgnn -de 3 -d ../data/GBA -lr 0.01 -e 400 -lp results/dc_gba.pt -s 0 -ned 32 -c 10
# python -m model.load_mtgnn -de 2 -d ../data/GLA -lr 0.01 -e 400 -b 128 -lp results/dc_gla_1e-2.pt -s 0 -ned 32
# python -m model.load_mtgnn -de 2 -d ../data/ERA5 -lr 0.01 -e 400 -b 128 -lp results/dc_dsa_cluster_era5_1e-3.pt -s 0 -ned 32
# python -m model.load_mtgnn -de 1 -d ../data/AURORA -lr 3e-4 -e 200 -b 32 -lp results/dc_aurora_1e-3.pt -s 0 -ned 32
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lr', '--lr',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-lp', '--load_path', type=str, default='results/')
    parser.add_argument('-ned', '--ne_dim',type=int,default=128,help='')
    parser.add_argument('-c', '--check', type=int, default=5, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    parser.add_argument('-a', '--ae', action='store_true')
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)

    raw_data = torch.load(args.load_path)
    test_syn(args, dataloader, raw_data, device)