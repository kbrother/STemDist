from tqdm import tqdm
from .fgnn import FGN
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


def test_fgnn(args, data, synx, syny, device):
    
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]
    seq_len = dataloader['train_loader'].xs.shape[1]
    _model = FGN(pre_length=seq_len, embed_size=args.embed_size, seq_length=seq_len, hidden_size=args.hidden_size)
    scaler = data['scaler']
    
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=args.lr)
    _model = _model.to(device)

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


# python -m model.fgnn.load_fgnn -de 5 -d ../data/GBA -lr 1e-2 -e 300 -lp results/random_gba.pt -b 32
# python -m model.fgnn.load_fgnn -de 7 -d ../data/GLA -lr 1e-2 -e 500 -lp results/random_gla.pt -b 32
# python -m model.fgnn.load_fgnn -de 7 -d ../data/ERA5 -lr 1e-2 -e 500 -lp results/random_era5.pt -b 32
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('-de', '--device', type=int, default=0, help='')
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**7, help='batch size')
    args.add_argument('-lr', '--lr', default=0.003, type=float)
    args.add_argument('-e', '--epochs', default=100, type=int)
    args.add_argument('-lp', '--load_path', type=str, default='results/') 
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
    args.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
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

    test_fgnn(args, dataloader, synx, syny, device)