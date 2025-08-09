import argparse
import util
from model.agcrn.agcrn import AGCRN
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import random
import torch
import numpy as np
import sys
import math
import copy


def test_syn(args, data, synx, syny, device):    
    scaler = dataloader['scaler']
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = synx.shape[3]    
    _model = AGCRN(args, num_nodes, in_dim)
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=args.lr)
    _model = _model.to(device)
    for p in _model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    min_val_loss = sys.float_info.max
    for i in tqdm(range(args.epochs)):
        _model.train()
        output_syn = _model(synx).squeeze()
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        _model.eval()
        if (i+1)%10 == 0:
            with torch.no_grad():
                val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
    
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
    with open(args.save_path, "a") as f:
        f.write(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}\n")


# python -m model.agcrn.load_agcrn_coreset -de 7 -d ../data/GBA -lr 0.01 -e 100 -b 32 -rr 0.01 -lp results/random_gba.pt
# python -m model.agcrn.load_agcrn_baseline -de 7 -d ../data/GLA -lr 0.001 -e 300 -b 32 -lp results/random_gla.pt
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-de', '--device', type=int, default=0, help='')
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-r', '--rnn_units', type=int, default=2**6, help='rnn hidden unit')
    args.add_argument('-nl', '--num_layers', default=2, type=int)
    args.add_argument('-lr', '--lr', default=0.003, type=float)
    args.add_argument('-e', '--epochs', default=100, type=int)
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args.add_argument('-lp', '--load_path', type=str, default='results/') 
    args.add_argument('-sp', '--save_path', type=str, default='results/')
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

    if (len(synx.shape) <=3):
        synx = synx.unsqueeze(-1)
        
    print("load finish")
    print(synx.shape)
    print(syny.shape)
    
    test_syn(args, dataloader, synx, syny, device)