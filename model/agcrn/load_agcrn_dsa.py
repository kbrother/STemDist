from tqdm import tqdm
from model.agcrn.agcrn import AGCRN
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


def test_agcrn(args, data, raw_data, device):
    synx = raw_data['x'].to(device)
    syny = raw_data['y'].to(device)
    _weight = raw_data['w'].to(device)
    
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]
    _model = AGCRN(args, num_nodes, in_dim, True, args.ne_dim)
    scaler = data['scaler']
    
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=args.lr)
    _model = _model.to(device)
    for p in _model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    nm_input_real = np.mean(data['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
    nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
    nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
    nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim
     
    nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
    nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
    nm_input_syn = torch.reshape(nm_input_syn, (synx.shape[2], -1))
    
    min_val_loss = sys.float_info.max
    num_sample = round(synx.shape[2]/4)
    for i in tqdm(range(args.epochs)):
        _model.train()
        _order = list(range(synx.shape[2]))
        random.shuffle(_order)

        for j in range(4):
            if j < 3:
                _idx = _order[num_sample*j:num_sample*(j+1)]
            else:
                _idx = _order[num_sample*j:]     
            curr_weight = _weight[:,:,_idx] / torch.sum(_weight[:,:,_idx]) * num_sample
            _model.embed_forward(nm_input_syn[_idx])
            
            output_syn = _model(synx[:,:,_idx,:]).squeeze()
            loss_syn = torch.square(output_syn - syny[:,:,_idx]) * curr_weight
            loss_syn = torch.mean(loss_syn)
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
    with open(args.save_path, "a") as f:
        f.write(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}\n")


# python -m model.agcrn.load_agcrn -de 7 -d ../data/GBA -lr 1e-3 -e 300 -lp results/dc_clus_gba.pt
# python -m model.agcrn.load_agcrn -de 7 -d ../data/GLA -lr 1e-3 -e 300 -lp results/dc_clus_gla.pt
# python -m model.agcrn.load_agcrn -de 7 -d ../data/ERA5 -lr 1e-3 -e 300 -lp results/dc_clus_era5.pt -b 32
if __name__ == "__main__":
    torch.set_num_threads(4)
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('-de', '--device', type=int, default=0, help='')
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**7, help='batch size')
    args.add_argument('-r', '--rnn_units', type=int, default=2**6, help='rnn hidden unit')
    args.add_argument('-nl', '--num_layers', default=2, type=int)
    args.add_argument('-lr', '--lr', default=0.003, type=float)
    args.add_argument('-e', '--epochs', default=100, type=int)
    args.add_argument('-lp', '--load_path', type=str, default='results/') 
    args.add_argument('-ned', '--ne_dim',type=int,default=32,help='')
    args.add_argument('-sp', '--save_path', type=str, default='results/')
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args = args.parse_args()    

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    raw_data = torch.load(args.load_path)


    test_agcrn(args, dataloader, raw_data, device)