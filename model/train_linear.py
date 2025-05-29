import argparse
import util
from model.linear import Linear
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import random
import torch
import numpy as np
import sys


# python -m model.train_linear -de 0 -lr 0.1 -e 30
# python -m model.train_linear -d ../data/PEMS-BAY -de 0 -lr 0.1 -e 100
# python -m model.train_linear -d ../data/ELECTRICITY -de 0 -lr 0.1 -e 100
# python -m model.train_linear -d ../data/SOLAR -de 0 -lr 0.1 -e 100
# python -m model.train_linear -d ../data/TRAFFIC -de 1 -lr 0.1 -e 100
if __name__ == "__main__":
    torch.set_num_threads(4)
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('-de', '--device', type=int, default=0, help='')
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-lr', '--lr', default=0.003, type=float)
    args.add_argument('-e', '--epochs', default=100, type=int)
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args = args.parse_args()    

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
    input_len = dataloader['train_loader'].xs.shape[1]
    model = Linear(args, input_len, input_len, in_dim)
    #init loss function, optimizer    

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    model = model.to(device)
    
    min_val_mse = sys.float_info.max
    for e in range(args.epochs):
        model.train()
        train_loss, num_entry = 0, 0
        dataloader['train_loader'].shuffle()
        
        for iter, (x, y) in enumerate(tqdm(dataloader['train_loader'].get_iterator())):
            trainx = torch.tensor(x, device=device, dtype=torch.float)
            trainy = torch.tensor(y, device=device, dtype=torch.float)
            output = model(trainx)
            output = scaler.inverse_transform(output)
            curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)            
            curr_loss /= num_val_entry

            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():               
            val_mse = model.test_model(dataloader['val_loader'], scaler, device)
            test_mse = model.test_model(dataloader['test_loader'], scaler, device)            
            print(f'epoch: {e}, valid mse: {val_mse}, test mse: {test_mse}')
