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
    if 'w' in raw_data:
        _weight = raw_data['w'].to(device)
    else:
        _weight = None
    print("load finish")
    print(synx.shape)
    print(syny.shape)

    scaler = data['scaler']
    num_nodes = data['train_loader'].xs_orig.shape[2]
    in_dim = data['train_loader'].xs.shape[3]    
    seq_len = data['train_loader'].xs.shape[1]    
    out_dim = 1
    _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim)      
    _model.to(device)
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=args.lr)
    min_val_loss = sys.float_info.max
   
    nm_input_real = np.mean(data['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
    nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
    nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
    nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim
    
    nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
    nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
    nm_input_syn = torch.reshape(nm_input_syn, (synx.shape[2], -1))
    
    num_sample = round(synx.shape[2]/4)
    for i in tqdm(range(args.epochs)):
        _model.train()
        for j in range(4):
            _idx = random.sample(range(synx.shape[2]), num_sample)
            curr_weight = _weight[:,:,_idx] / torch.sum(_weight[:,:,_idx]) * num_sample
            _model.embed_forward(nm_input_syn[_idx])
            output_syn = _model(synx[:,:,_idx,:].transpose(1, 3)).squeeze()
            loss_syn = torch.square(output_syn - syny[:,:,_idx]) * curr_weight
            loss_syn = torch.mean(loss_syn)
                #loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()
        
        _model.eval()
        if (i+1)%5 == 0:
            with torch.no_grad():
                _model.embed_forward(nm_input_real)
                val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
                test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))
    
            print(f"epoch :{i}, train loss: {loss_syn}, val loss: {val_loss}, test loss: {test_loss}")
            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():    
        _model.embed_forward(nm_input_real)
        test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))
    print(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")      


# python -m model.load_mtgnn_dsa -de 0 -d ../data/GBA -lr 0.001 -e 400 -b 128 -lp results/dc_dsa_cluster_gba_1e-3.pt -s 0 -ned 32
# python -m model.load_mtgnn_dsa -de 2 -d ../data/GLA -lr 0.001 -e 400 -b 128 -lp results/dc_dsa_cluster_gla_1e-3.pt -s 0 -ned 32
# python -m model.load_mtgnn_dsa -de 1 -d ../data/ERA5 -lr 0.001 -e 200 -b 128 -lp results/dc_dsa_cluster_era5_1e-3.pt -s 0 -ned 32
# python -m model.load_mtgnn_dsa -de 0 -d ../data/AURORA -lr 1e-3 -e 100 -b 64 -lp results/dc_dsa_cluster_aurora_1e-3_1e-3_final.pt -s 0 -ned 32
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
    parser.add_argument('-ned', '--ne_dim',type=int,default=32,help='')
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