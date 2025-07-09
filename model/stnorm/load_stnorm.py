from tqdm import tqdm
from model.stnorm.stnorm import STNorm
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


def test_stnorm(args, data, synx, syny, device):

    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    scaler = data['scaler']
    seq_len = data['train_loader'].xs.shape[1]
    _model =  STNorm(device, True, True, in_dim, seq_len, 16, 2, 1, 4)

    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
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
    
            print(f"epoch :{i}, train loss: {loss_syn.item()} val loss: {val_loss}")
            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():
        test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))

    print(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")


# python -m model.stnorm.load_stnorm -de 5 -d ../data/GBA -lrs 1e-3 -lp results/dc_clus_gba.pt -b 32
# python -m model.stnorm.load_stnorm -de 5 -d ../data/GBA -lrs 1e-3 -lp results/random_gba.pt -b 32
# python -m model.stnorm.load_stnorm -de 7 -d ../data/GLA -lrs 1e-3 -lp results/dc_clus_gla.pt -e 300 -b 32
# python -m model.stnorm.load_stnorm -de 7 -d ../data/ERA5 -lrs 1e-3 -lp results/dc_clus_era5.pt -e 300 -b 32
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**5, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')    
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-lp', '--load_path', type=str, default='results/') 
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
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

    print(synx.shape)
    print(syny.shape)

    test_stnorm(args, dataloader, synx, syny, device)