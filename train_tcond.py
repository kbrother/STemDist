import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from model import *
import torch.optim as optim
import math
from tcond import TCond


# python train_tcond.py -lr 0.01 -rr 0.001 -e 1000 -sp results/metr-la.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=2**12, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('-rr', '--reduction_rate',type=float,default=0.005,help='learning rate')
    parser.add_argument('-dr', '--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('-wd', '--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str)
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)

    _tcond = TCond(dataloader, args, device)
    _tcond.train()