from tqdm import tqdm
from model.mtgnn import gtnet
from model.dcrnn.dcrnn_model import DCRNNModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import argparse
import numpy as np
import util
import sys
import copy
import math


def test_mtgnn(args, data, synx, syny, device):
    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    seq_len = data['train_loader'].xs.shape[1]
    scaler = data['scaler']
    out_dim = 1

    _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim)     
    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_mtgnn)
    nm_input_real = np.mean(data['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
    nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
    nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
    nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim
     
    nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
    nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
    nm_input_syn = torch.reshape(nm_input_syn, (synx.shape[2], -1))

    min_val_loss = sys.float_info.max
    for i in tqdm(range(args.epochs)):
        _model.train()
        _model.embed_forward(nm_input_syn)
        output_syn = _model(synx.transpose(1,3)).squeeze()
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%10 == 0:
            with torch.no_grad():
                _model.embed_forward(nm_input_real)
                val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
    
            print(f"epoch :{i}, val loss: {val_loss}")
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

    with torch.no_grad():
        idx = torch.arange(nm_input_real.shape[0]).to(device)
        adj_large = _model.gc(idx)
        _model.embed_forward(nm_input_syn)
        
        idx = torch.arange(nm_input_syn.shape[0]).to(device)
        adj_small = _model.gc(idx)

    return adj_small.cpu().numpy(), adj_large.cpu().numpy()


def test_dcrnn(args, data, synx, syny, device, adj_small, adj_large):
    num_nodes_small = synx.shape[2]
    num_nodes_large = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    seq_len = data['train_loader'].xs.shape[1]
    scaler = data['scaler']
    model_kwargs = {
        "cl_decay_steps": 2000, "filter_type": "dual_random_walk",
        "horizon": 12, "input_dim": 2, "l1_decay": 0,
        "max_diffusion_step": 2,
        "num_nodes_small": num_nodes_small,
        "num_nodes_large": num_nodes_large,
        "num_rnn_layers": 2, "output_dim": 1,
        "rnn_units": 64, "seq_len": 12
    }

    print(num_nodes_small)
    _model = DCRNNModel(adj_small, adj_large, **model_kwargs)
    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_dcrnn)

    min_val_loss = sys.float_info.max

    batch_size = synx.shape[0]
    synx = torch.transpose(synx, 0, 1)
    synx = torch.reshape(synx, (seq_len, batch_size, -1))
    syny = torch.transpose(syny, 0, 1)
    syny = torch.reshape(syny, (seq_len, batch_size, -1))
    
    for i in tqdm(range(args.epochs)):
        _model.train()
        output_syn = _model(synx, True)
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        print(f"epoch: {i}, loss: {loss_syn}")
        optimizer.step()

        _model.eval()
        if (i+1)%10 == 0:
            with torch.no_grad():
                val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
    
            print(f"epoch :{i}, val loss: {val_loss}")
            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():
        test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))

    print(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")
    

# python -m model.dcrnn.load_dcrnn -de 1 -d ../data/GBA -lrm 1e-2 -lrd 1e-3 -e 10 -lp results/dc_clus_gba.pt
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('-de', '--device', type=int, default=0, help='')
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-lrm', '--lr_mtgnn', default=0.003, type=float)
    args.add_argument('-lrd', '--lr_dcrnn', default=0.003, type=float)
    args.add_argument('-e', '--epochs', default=100, type=int)
    args.add_argument('-lp', '--load_path', type=str, default='results/') 
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args.add_argument('-ned', '--ne_dim',type=int,default=128,help='')
    args = args.parse_args()

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

    print(synx.shape)
    print(syny.shape)

    adj_small, adj_large = test_mtgnn(args, dataloader, synx, syny, device)
    print(np.sum(np.sum(adj_small, axis=1) == 0))
    print(np.sum(np.sum(adj_large, axis=1)==0))
    test_dcrnn(args, dataloader, synx, syny, device, adj_small, adj_large)
    