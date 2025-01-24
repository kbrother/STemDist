from tqdm import tqdm
from model.gwave import gwnet
from model.grnn import GRNN
from model.mtgnn import gtnet
from model.stemgnn import StemGNN
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import argparse
import numpy as np
import util
import sys
import copy
from model.agcrn import AGCRN


def test_gwnet(args, data, synx, syny, node_embed1, node_embed2, device):
    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    scaler = data['scaler']
    _model = gwnet(device, num_nodes, args.dropout, in_dim, args.seq_length, 
                   residual_channels=args.nhid, dilation_channels=args.nhid, 
                   skip_channels=8*args.nhid, end_channels=16*args.nhid, node_vec1=node_embed1, node_vec2=node_embed2)
    ''' 
    _model = gwnet(device, num_nodes, args.dropout, in_dim, args.seq_length, 
                   residual_channels=args.nhid, dilation_channels=args.nhid, 
                   skip_channels=8*args.nhid, end_channels=16*args.nhid)
    '''
    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
    min_val_loss = sys.float_info.max
    for i in tqdm(range(400)):
        _model.train()
        output_syn = _model(synx.transpose(1,3)).squeeze()
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%10 == 0:
            with torch.no_grad():
                val_loss = _model.test_model(data['val_loader'], scaler)

            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())
            print(f'min i: {min_i}, val: {val_loss}')

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():
        test_loss = _model.test_model(data['test_loader'], scaler)

    print(f'test :{test_loss}')


def test_agcrn(args, data, synx, syny, node_embed, device):
    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    scaler = data['scaler']
    _model = AGCRN(args, num_nodes, in_dim, node_embed)
    #_model = AGCRN(args, num_nodes, in_dim)
    for p in _model.parameters():
        if p.shape == (num_nodes, args.embed_dim):
            print("here")
            continue
            
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)   

    
    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
    min_val_loss = sys.float_info.max
    for i in tqdm(range(400)):
        _model.train()
        output_syn = _model(synx).squeeze()
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%10 == 0:
            with torch.no_grad():
                val_loss = _model.test_model(data['val_loader'], scaler, device)

            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())
            print(f'i: {i}, val: {val_loss}')

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():
        test_loss = _model.test_model(data['test_loader'], scaler, device)

    print(f'test :{test_loss}')    


def test_mtgnn(args, data, synx, syny, node_embed, device):
    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    scaler = data['scaler']
    _model = gtnet(True, True, 2, num_nodes,
                  device, predefined_A=None, static_feat=node_embed.to(device),
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,
                  conv_channels=32, residual_channels=32,
                  skip_channels=64, end_channels=128,
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)    
    
    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
    min_val_loss = sys.float_info.max
    for i in tqdm(range(300)):
        _model.train()
        output_syn = _model(synx.transpose(1,3)).squeeze()
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%10 == 0:
            with torch.no_grad():
                val_loss = _model.test_model(data['val_loader'], scaler, device)

            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())
            print(f'min i: {min_i}, val: {val_loss}')

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():
        test_loss = _model.test_model(data['test_loader'], scaler, device)

    print(f'test :{test_loss}')


def test_stemgnn(args, data, synx, syny, node_embed, device):
    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    scaler = data['scaler']
    _model = StemGNN(num_nodes, 2, 12, args.multi_layer, horizon=12)    
    
    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
    min_val_loss = sys.float_info.max
    for i in tqdm(range(300)):
        _model.train()
        output_syn, _ = _model(synx[:,:,:,0])
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%10 == 0:
            with torch.no_grad():
                val_loss = _model.test_model(data['val_loader'], scaler, device)

            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())
            print(f'min i: {min_i}, val: {val_loss}')

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():
        test_loss = _model.test_model(data['test_loader'], scaler, device)

    print(f'test :{test_loss}')


# python -m load_and_train -de 1 -lrs 1e-3 -lp results/dc_mtgnn.pt
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**11, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')    
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-lp', '--load_path', type=str, default='results/') 
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-dr', '--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    parser.add_argument('-r', '--rnn_units', type=int, default=2**6, help='rnn hidden unit')
    parser.add_argument('-nl', '--num_layers', default=2, type=int)
    parser.add_argument('-ed', '--embed_dim', default=10, type=int)
    parser.add_argument('--multi_layer', type=int, default=5)
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
    node_embed1 = raw_data['node1']
    node_embed2 = raw_data['node2']

    print(synx.shape)
    print(syny.shape)
    
    #test_gwnet(args, dataloader, synx, syny, node_embed1, node_embed2, device)
    #test_agcrn(args, dataloader, synx, syny, node_embed1, device)
    test_stemgnn(args, dataloader, synx, syny, node_embed1, device)
    #test_mtgnn(args, dataloader, synx, syny, node_embed, device)