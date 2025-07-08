import torch.nn as nn
import torch
import copy
import sys
import util
import random
from model.stgcn.stgcn import STGCN
from tqdm import tqdm
import numpy as np
import argparse
import math
import time
import torch.nn.functional as F


def test_syn(args, data, device):
    num_total = data['train_loader'].xs.shape[0]
    num_elems = round(args.reduction_rate * num_total)
    sampled_idx = random.sample(list(range(num_total)), num_elems)
    synx = data['train_loader'].xs[sampled_idx]
    syny = data['train_loader'].ys[sampled_idx]
    synx = torch.tensor(synx, device=device, dtype=torch.float)
    syny = torch.tensor(syny, device=device, dtype=torch.float)
    syny = data['scaler'].transform(syny)
    print(f'x: {synx.shape}, y:{syny.shape}')
    
    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    seq_len = data['train_loader'].xs.shape[1]
    scaler = data['scaler']
    
    _model = STGCN(in_dim, seq_len, seq_len, 128)  
    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.learning_rate)
    min_val_loss = sys.float_info.max

    nm_input_real = np.mean(data['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
    nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
    nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
    nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim
     
    nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
    nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
    nm_input_syn = torch.reshape(nm_input_syn, (synx.shape[2], -1))
   
    synx = torch.transpose(synx, 1, 2)
    syny = torch.transpose(syny, 1, 2)
    for i in tqdm(range(args.epochs)):
        _model.train()
        _model.embed_forward(nm_input_syn)
        output_syn = _model(synx)
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%10 == 0:
            with torch.no_grad():
                _model.embed_forward(nm_input_real)
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
        test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))
    print(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")       




# python -m model.stgcn.random_sample_stgcn -d ../data/GBA -de 0 -s 0 -lr 1e-3 -r 1e-2
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-r', '--reduction_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=2**7, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-e', '--epochs', default=100, type=int)
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")
    test_syn(args, dataloader, device)