import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model import *
import torch.optim as optim
import math
from gwave_grad import GwaveGrad
from core_set import *
from gwave_grad_clus import GwaveGradClus


# python gwave/train.py ours_gc -lr 0.01 -rr 0.001 -e 1000 -sp results/oursgc_metrc-la_lr0.001.txt
# python gwave/train.py ours_g -lr 0.1 -rr 0.001 -e 1000 -sp results/oursg_metrc-la_lr0.1.txt
# python gwave/train.py random -rr 0.001 -e 100 -sp results/random_metr-la_rr0.001.txt -de 3
# python gwave/train.py kmeans -rr 0.001 -e 100 -sp results/kmeans_metr-la_rr0.001.txt -de 0 -b 64
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, help='which algorithm?')
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-4,help='learning rate')
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

    if args.agent == "ours_g":
        _tcond = GwaveGrad(dataloader, args, device)
    elif args.agent == "ours_gc":
        _tcond = GwaveGradClus(dataloader, args, device)
    elif args.agent == "random":
        _tcond = RandomSample(dataloader, args, device)
    elif args.agent == "kmeans":
        _tcond = Kmeans(dataloader, args, device)
    _tcond.train()