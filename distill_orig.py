import torch.nn as nn
from tqdm import tqdm
from model.mtgnn import gtnet
import torch
import util
import sys
import copy
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import copy
import math


class DataDistill:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_series = int(args.reduce_rate * data['train_loader'].xs.shape[0])
        scaler = data['scaler']
        
        # Define condensed data
        # Define condensed data
        num_series_total = data['train_loader'].xs.shape[0]
        num_nodes_total = data['train_loader'].xs.shape[2]
        
        sampled_idx1 = random.sample(list(range(num_series_total)), self.num_series)
        sampled_idx1.sort()        
        self.synx = self.data['train_loader'].xs[sampled_idx1]     
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        
        self.syny = self.data['train_loader'].ys[sampled_idx1]
        self.syny = scaler.transform(self.syny)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
        
        self.synx = nn.Parameter(self.synx)
        self.syny = nn.Parameter(self.syny)
        print(f'feat x shape: {self.synx.shape}')
        print(f'feat y shape: {self.syny.shape}')

        num_iter = 3
        val_sum, test_sum = 0, 0
        for i in range(num_iter):
            min_i, val_loss, test_loss = self.test_syn()
            print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
            val_sum += val_loss
            test_sum += test_loss
        with open(args.save_path + ".txt", 'a') as f:
            f.write(f"initial, min i: {min_i}, val loss: {val_sum/num_iter}, test loss: {test_sum/num_iter}\n")   
        
        
    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx.detach().clone()
        syny = self.syny.detach().clone()
        
        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        seq_len = data['train_loader'].xs.shape[1]
        scaler = data['scaler']
        out_dim = 1

        _model = gtnet(True, True, 2, num_nodes, 
                  self.device, predefined_A=None, use_static_feat=False,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
        min_val_loss = sys.float_info.max              
        for i in tqdm(range(args.check_epoch)):  
            _model.train()            
            output_syn = _model(synx.transpose(1,3)).squeeze()
            loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():                    
                    val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, self.device))
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():            
            test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, self.device))

        return min_i, min_val_loss, test_loss