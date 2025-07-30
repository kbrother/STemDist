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


def test_syn(args, data, synx, syny, device):    
    scaler = dataloader['scaler']
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]    
    seq_len = dataloader['train_loader'].xs.shape[1]    
    out_dim = 1
    _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=False,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)      
    _model.to(device)
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=args.lr)
    _model = _model.to(device)
   
    min_val_loss = sys.float_info.max
    for i in tqdm(range(args.epochs)):
        _model.train()
        output_syn = _model(synx.transpose(1, 3)).squeeze()
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%5 == 0:
            with torch.no_grad():
                val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
                #test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))
            print(f"epoch :{i}, train loss: {loss_syn}, val loss: {val_loss}")
            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():        
        test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))
    print(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")      


# python -m model.load_mtgnn_baseline -de 0 -d ../data/GBA -lr 1e-4 -e 400 -b 128 -lp results/random_gba_0.pt -s 0
# python -m model.load_mtgnn_baseline -de 0 -d ../data/GLA -lr 1e-3 -e 400 -b 128 -lp results/random_gla_0.pt -s 3
# python -m model.load_mtgnn_baseline -de 4 -d ../data/ERA5 -lr 1e-3 -e 100 -b 128 -lp results/random_era5_0.pt -s 0
# python -m model.load_mtgnn_baseline -de 0 -d ../data/CA -lr 1e-2 -e 100 -b 128 -lp results/random_ca_0.pt -s 0
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
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)

    raw_data = torch.load(args.load_path)
    synx = raw_data['x'].to(device)
    syny = raw_data['y'].to(device)
    print("load finish")
    print(synx.shape)
    print(syny.shape)

    test_syn(args, dataloader, synx, syny, device)