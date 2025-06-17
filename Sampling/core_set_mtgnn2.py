import torch.nn as nn
import torch
import copy
import sys
import util
import random
from model.mtgnn import gtnet
from tqdm import tqdm
import numpy as np
import argparse
import math
import torch.nn.functional as F
import time


class Coreset:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_series = int(args.series_reduce_rate * data['train_loader'].xs.shape[0])
        self.num_nodes = int(args.node_reduce_rate * data['train_loader'].xs.shape[2])


    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx
        syny = self.syny

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        seq_len = data['train_loader'].xs.shape[1]
        scaler = data['scaler']
        if len(dataloader['train_loader'].ys.shape) == 4:        
            out_dim = dataloader['train_loader'].ys.shape[3]
        else: 
            out_dim = 1
        _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim)     
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.learning_rate)
        min_val_loss = sys.float_info.max

        nm_input_real = np.mean(dataloader['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
        nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
        nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
        nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim
         
        nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
        nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
        nm_input_syn = torch.reshape(nm_input_syn, (self.num_nodes, -1))
        time_sum = 0
        for i in tqdm(range(200)):  
            start_time = time.time()
            _model.train()
            _model.embed_forward(nm_input_syn)
            output_syn = _model(synx.transpose(1,3)).squeeze()
            loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()
            time_sum += time.time() - start_time

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():
                    _model.embed_forward(nm_input_real)
                    val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            _model.embed_forward(nm_input_real)
            test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))

        return min_i, min_val_loss, test_loss, time_sum


class RandomSample(Coreset):
    def __init__(self, data, args, device):
        super().__init__(data, args, device)
        
        num_series_total = data['train_loader'].xs.shape[0]
        num_nodes_total = data['train_loader'].xs.shape[2]
        num_seq = data['train_loader'].xs.shape[1]
        num_feat = data['train_loader'].xs.shape[3]
                
        sampled_idx1 = random.sample(list(range(num_series_total)), self.num_series)
        sampled_idx1.sort()
        sampled_idx2 = random.sample(list(range(num_nodes_total)), self.num_nodes)
        sampled_idx2.sort()
        self.synx = self.data['train_loader'].xs[sampled_idx1][:,:,sampled_idx2,:]     
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        
        self.syny = self.data['train_loader'].ys[sampled_idx1][:,:,sampled_idx2]
        #self.syny = data['scaler'].transform(self.syny)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
        
        print(f'x: {self.synx.shape}, y:{self.syny.shape}')


# python -m Sampling.core_set_mtgnn2 -d ../data/METR-LA -de 1 -s 0 -lr 1e-3 -srr 1e-2 -nrr 2e-1 -ned 32
# python -m Sampling.core_set_mtgnn2 -d ../data/PEMS-BAY -de 0 -s 0 -lr 1e-2 -srr 1e-2 -nrr 0.2 -ned 128
# python -m Sampling.core_set_mtgnn2 -d ../data/AIR-DATA -de 0 -s 0 -lr 1e-2 -srr 1e-2 -nrr 0.2 -ned 32
# python -m Sampling.core_set_mtgnn2 -d ../data/ELECTRICITY -de 1 -s 0 -lr 1e-3 -srr 1e-2 -nrr 2e-1 -ned 16
# python -m Sampling.core_set_mtgnn2 -d ../data/SOLAR -de 7 -lr 1e-3 -srr 1e-2 -nrr 2e-1 -ned 32
# python -m Sampling.core_set_mtgnn2 -d ../data/TRAFFIC -de 1 -lr 1e-3 -srr 1e-2 -nrr 2e-1 -ned 64
# python -m Sampling.core_set_mtgnn2 -d ../data/PEMS07 -de 0 -lr 1e-3 -srr 1e-2 -nrr 2e-1 -ned 32
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-nrr', '--node_reduce_rate',type=float,default=1,help='learning rate')
    parser.add_argument('-srr', '--series_reduce_rate',type=float,default=2e-2,help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-ned', '--ne_dim',type=int,default=128,help='')
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")

    _model = RandomSample(dataloader, args, device)
    min_i, min_val_loss, test_loss, time_sum = _model.test_syn()
    mem_use = torch.cuda.memory_allocated(device)/10**6
    print(f"min i: {min_i}, min val loss: {min_val_loss}, test loss: {test_loss}, time sum: {time_sum}, mem: {mem_use}")