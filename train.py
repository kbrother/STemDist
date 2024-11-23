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


# python train.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=2**10, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('-dr', '--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('-wd', '--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")
    
    scaler = dataloader['scaler']
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]
    model = gwnet(device, num_nodes, args.dropout, in_dim, args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=8*args.nhid, end_channels=16*args.nhid)
    model.to(device)
    
    _optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for i in tqdm(range(args.epochs)):

        model.train()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.tensor(x, device=device, dtype=torch.float)
            trainx= trainx.transpose(1, 3)
            trainy = torch.tensor(y, device=device, dtype=torch.float)
            trainy = trainy[:,:,:,0]
            output = model(trainx).squeeze()
            output = scaler.inverse_transform(output)
            curr_loss = util.mae(output, trainy)

            _optimizer.zero_grad()
            curr_loss.backward()
            _optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                valx = torch.tensor(x, device=device, dtype=torch.float)
                valx = valx.transpose(1, 3)
                valy = torch.tensor(y, device=device, dtype=torch.float)
                valy = valy[:,:,:,0]
                output = model(valx).squeeze()
                output = scaler.inverse_transform(output)
                val_loss += util.ae(output, valy).item()

            test_loss = 0
            for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                testx = torch.tensor(x, device=device, dtype=torch.float)
                testx = testx.transpose(1, 3)
                testy = torch.tensor(y, device=device, dtype=torch.float)
                testy = testy[:,:,:,0]
                output = model(testx).squeeze()
                output = scaler.inverse_transform(output)
                test_loss += util.ae(output, testy).item()

            print(f'epoch: {i}, valid mae: {val_loss/math.prod(dataloader["val_loader"].ys.shape[:-1])}, test mae: {test_loss/math.prod(dataloader["test_loader"].ys.shape[:-1])}')