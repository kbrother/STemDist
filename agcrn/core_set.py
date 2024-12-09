import torch.nn as nn
import torch
import copy
import sys
import utils
import random
from tqdm import tqdm
import numpy as np
from model import AGCRN
import torch.nn.functional as F
from train_agcrn import test_model


class Coreset:

    def __init__(self, loader_list, args, device, scaler):
        self.train_loader = loader_list[0]
        self.val_loader = loader_list[1]
        self.test_loader = loader_list[2]
        self.args = args
        self.device = device
        self.scaler = scaler
        self.num_elems = int(args.reduction_rate * self.train_loader.dataset.tensors[0].shape[0])
        print(f"reduced elements: {self.num_elems}")
    

    def test_syn(self):
        args = self.args
        synx = self.synx
        syny = self.syny

        num_nodes = synx.shape[2]
        input_dim = synx.shape[3]
        model = AGCRN(args, num_nodes, input_dim)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)        
        model.to(self.device)    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            model.train()
            output = model(self.synx).squeeze()
            curr_loss = F.l1_loss(output, self.syny[...,0])

            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()

            if (i+1)%5 == 0:
                val_loss = test_model(model, self.val_loader, self.scaler, self.device)
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(model.state_dict())
                    min_i = i
    
                print(f'i: {i}, val_loss: {val_loss}')
                
        model.load_state_dict(min_params)
        test_loss = test_model(model, self.test_loader, self.scaler, self.device)
        return min_i, min_val_loss, test_loss


class RandomSample(Coreset):
    def __init__(self, loader_list, args, device, scaler):
        super().__init__(loader_list, args, device, scaler)

        self.args = args
        x_total = self.train_loader.dataset.tensors[0]
        y_total = self.train_loader.dataset.tensors[1]
        num_total = x_total.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = x_total[sampled_idx]
        self.syny = y_total[sampled_idx]
        self.synx = self.synx.to(device)
        self.syny = self.syny.to(device)
    

    def train(self):
        min_i, val_loss, test_loss = self.test_syn()
        print(f"min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
     #  with open(self.args.save_path, 'a') as f:
     #      f.write(f"min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        
