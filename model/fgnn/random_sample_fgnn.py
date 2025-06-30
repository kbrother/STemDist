import torch.nn as nn
import torch
import copy
import sys
import util
import random
from .fgnn import FGN
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
    if len(dataloader['train_loader'].ys.shape) == 4:        
        out_dim = dataloader['train_loader'].ys.shape[3]
    else: 
        out_dim = 1
    
    _model = FGN(pre_length=seq_len, embed_size=args.embed_size, seq_length=seq_len, hidden_size=args.hidden_size)  
    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.learning_rate)
    min_val_loss = sys.float_info.max

    min_val_loss = sys.float_info.max
    synx = synx[:,:,:,0]
    for i in tqdm(range(args.epochs)):
        _model.train()
        output_syn = _model(synx).squeeze()
        output_syn = torch.transpose(output_syn, 2, 1)
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





# python -m model.fgnn.random_sample_fgnn -d ../data/GBA -de 0 -s 0 -lr 1e-3 -r 1e-2
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-r', '--reduction_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=2**7, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
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