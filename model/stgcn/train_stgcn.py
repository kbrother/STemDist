import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.stgcn.stgcn import STGCN
import torch.optim as optim
import math
from scipy.io import loadmat
import sys
import numpy as np


# python -m model.stgcn.train_stgcn -de 0 -d ../data/GBA -lr 2e-3 -e 200
# python -m model.stgcn.train_stgcn -de 0 -d ../data/GLA -lrs 2e-3 -lp results/dc_clus_gla.pt
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**5, help='batch size')
    parser.add_argument('-lr', '--lr',type=float,default=1e-2,help='learning rate')    
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
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
    
    scaler = dataloader['scaler']
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]
    seq_len = dataloader['train_loader'].xs.shape[1]
    _model = STGCN(in_dim, seq_len, seq_len, 128)
    #model.reset_parameters()

    _model.to(device)
    _optimizer = optim.Adam(_model.parameters(), lr=args.lr)
    min_val_mse = sys.float_info.max
    
    nm_input_real = np.mean(dataloader['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
    nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
    nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
    nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim    
    for i in range(args.epochs):
        _model.train()        
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            _model.embed_forward(nm_input_real)
            trainx = torch.tensor(x, device=device, dtype=torch.float)
            trainx = trainx.transpose(1, 2)
            trainy = torch.tensor(y, device=device, dtype=torch.float)
            output = _model(trainx).transpose(1, 2)
            output = scaler.inverse_transform(output)
            curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
            curr_loss /= num_val_entry

            _optimizer.zero_grad()
            curr_loss.backward()
            _optimizer.step()

        _model.eval()
        with torch.no_grad():               
            val_mae = math.sqrt(_model.test_model(dataloader['val_loader'], scaler, device))
            test_mae = math.sqrt(_model.test_model(dataloader['test_loader'], scaler, device)         )   
            print(f'epoch: {i}, valid mae: {val_mae}, test mae: {test_mae}')