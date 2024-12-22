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
from gwave_traj import GwaveTraj
from gwave_grad_clus import GwaveGradClus


# python gwave/train.py ours_gc -de 3 -lr 0.01 -rr 0.001 -e 1000 -b 64 -sp results/oursgc_metrc-la_low-batch.txt
# python gwave/train.py ours_g -lr 0.1 -rr 0.001 -e 1000 -sp results/oursg_metrc-la_lr0.1.txt
# python gwave/train.py random -rr 0.001 -e 100 -sp results/random_metr-la_rr0.001.txt -de 3
# python gwave/train.py kmeans -rr 0.001 -e 100 -sp results/kmeans_metr-la_rr0.001.txt -de 0 -b 64
# python gwave/train.py ours_t -de 7 -lr 0.1 -lrl 1e-5 -e 100 -sp results/ourst_metrc-la.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, help='which algorithm?')
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-p', '--params', type=str, default='../data/params/METR-LA/')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1000,help='learning rate')
    parser.add_argument('-lrs', '--lr_student',type=float,default=1e-4,help='learning rate')
    parser.add_argument('-lrl', '--lr_lr',type=float,default=1e-7,help='learning rate')
    parser.add_argument('-rr', '--reduction_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('-dr', '--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('-wd', '--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-mse', '--max_start_epoch',type=int,default=25,help='')
    parser.add_argument('-ee', '--expert_epoch',type=int,default=3,help='')
    parser.add_argument('-ne', '--num_experts',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-ss', '--syn_steps', type=int, default=21, help='')
    parser.add_argument('-sp', '--save_path', type=str)
    parser.add_argument('-mp', '--mapping_path', type=str, default='mapping/METR-LA.npy')
    parser.add_argument('-nc', '--num_clusters', type=int, default=5, help='')
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
    elif args.agent == "ours_t":
        _tcond = GwaveTraj(dataloader, args, device)
    elif args.agent == "random":
        _tcond = RandomSample(dataloader, args, device)
    elif args.agent == "kmeans":
        _tcond = Kmeans(dataloader, args, device)
    _tcond.train()