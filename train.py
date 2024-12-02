from utils import *
import argparse
import random
import numpy as np
import torch
from core_set import *
from agcrn_grad import AgcrnGrad

# python train.py ours_g -de 3 -lr 0.1 -e 1000 -sp results/agcrn_ours_init_lr0.01.txt
# python train.py random -de 6 -sp results/agcrn_random.txt
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('agent', type=str, help='which algorithm?')
    args.add_argument('-de', '--device', type=int, default=0, help='')
    args.add_argument('-d', '--dataset', type=str, default='PEMS04', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-r', '--rnn_units', type=int, default=2**6, help='rnn hidden unit')
    args.add_argument('-nl', '--num_layers', default=2, type=int)
    args.add_argument('-ed', '--embed_dim', default=10, type=int)
    args.add_argument('-lr', '--lr', default=0.003, type=float)
    args.add_argument('-e', '--epochs', default=100, type=int)
    args.add_argument('-rr', '--reduction_rate',type=float,default=0.005,help='learning rate')
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args.add_argument('-sp', '--save_path', type=str, default='results/', help='data path')
    args = args.parse_args()   
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, val_loader, test_loader, scaler, _data = get_dataloader(args)
    device = torch.device(f"cuda:{args.device}")

    if args.agent == "random":
        _tcond = RandomSample([train_loader, val_loader, test_loader], args, device, scaler)
    elif args.agent == "ours_g":
        _tcond = AgcrnGrad(args,[train_loader, val_loader, test_loader], device, scaler)
    _tcond.train()