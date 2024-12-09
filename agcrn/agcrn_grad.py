import os
# OMP_NUM_THREADS: openmp, OPENBLAS_NUM_THREADS: openblas, MKL_NUM_THREADS: mkl, VECLIB_MAXIMUM_THREADS: accelerate, NUMEXPR_NUM_THREADS: numexpr
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
# os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=4

import torch.nn as nn
from tqdm import tqdm
import torch
torch.set_num_threads(4)
import utils
import sys
import copy
import random
from train_agcrn import test_model
from model import AGCRN
import torch.nn.functional as F


def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p) 

    
class AgcrnGrad:

    def __init__(self, args, loader_list, device, scaler):
        self.train_loader = loader_list[0]
        self.val_loader = loader_list[1]
        self.test_loader = loader_list[2]
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate * self.train_loader.dataset.tensors[0].shape[0])
        self.scaler = scaler
        
        # Define condensed data
        '''
        num_total = self.train_loader.dataset.tensors[0].shape[0]
        _shape = list(self.train_loader.dataset.tensors[0].shape[1:])
        _shape = [self.num_elems] + _shape
        self.synx = torch.rand(tuple(_shape), device=device, dtype=torch.float)
        
        _shape = list(self.train_loader.dataset.tensors[0].shape[1:-1])
        _shape = [self.num_elems] + _shape
        self.syny = torch.rand(tuple(_shape), device=device, dtype=torch.float)
        '''
        
        x_total = self.train_loader.dataset.tensors[0]
        y_total = self.train_loader.dataset.tensors[1] 
        num_total = x_total.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = x_total[sampled_idx]
        self.syny = y_total[sampled_idx,:,:,0]
        self.synx = self.synx.to(device)
        self.syny = self.syny.to(device)
        self.synx = nn.Parameter(self.synx)
        self.syny = nn.Parameter(self.syny)
        
        print(f'feat x shape: {self.synx.shape}')
        print(f'feat y shape: {self.syny.shape}')       


    def test_syn(self):
        args = self.args
        synx = self.synx.data.detach().clone()
        syny = self.syny.data.detach().clone()

        num_nodes = synx.shape[2]
        input_dim = synx.shape[3]
        model = AGCRN(args, num_nodes, input_dim)
        init_model(model)       
        model.to(self.device)    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            model.train()
            output = model(synx).squeeze()
            curr_loss = F.l1_loss(output, syny)

            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()

            if (i+1)%10 == 0:
                val_loss = test_model(model, self.val_loader, self.scaler, self.device)
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(model.state_dict())
                    min_i = i
                
        model.load_state_dict(min_params)
        test_loss = test_model(model, self.test_loader, self.scaler, self.device)
        return min_i, min_val_loss, test_loss        


    def train(self):
        args = self.args
        synx, syny = self.synx, self.syny

        num_nodes = synx.shape[2]
        input_dim = synx.shape[3]

        min_i, val_loss, test_loss = self.test_syn()
        print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        with open(args.save_path, 'a') as f:
            f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n") 

        optimizer = torch.optim.Adam([synx, syny], lr=args.lr)
        for i in range(args.epochs):
            model = AGCRN(args, num_nodes, input_dim)
            model_params = list(model.parameters())
            init_model(model)   
            model.to(self.device)
            model.train()
            optimizer_model = torch.optim.Adam(model_params, lr=0.01)
            
            train_loss = 0
            num_ol = 20
            for ol in tqdm(range(num_ol)):
                x, y = next(iter(self.train_loader))
                # comptue real gradient
                x = torch.stack(tuple(x), dim=0).to(self.device)
                y = torch.stack(tuple(y), dim=0).to(self.device)
                y = y[..., 0]
                output_real = model(x).squeeze()
                loss_real = F.l1_loss(output_real, y)
                gw_real = torch.autograd.grad(loss_real, model_params)
                gw_real = list((_.detach().clone() for _ in gw_real))
                
                # compute syntehtic graph                
                output_syn = model(synx).squeeze()
                loss_syn = F.l1_loss(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)
                
                _loss = utils.match_loss(gw_syn, gw_real, self.device)
                train_loss += loss_real.item()
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                if ol == num_ol-1:
                    break

                num_il = 5
                synx_inner, syny_inner = synx.detach(), syny.detach()
                for il in range(num_il):
                    optimizer_model.zero_grad()
                    output_syn_inner = model(synx_inner).squeeze()
                    loss_syn_inner = F.l1_loss(output_syn_inner, syny_inner)
                    loss_syn_inner.backward()
                    optimizer_model.step()
                
            train_loss /= num_ol
            if (i+1) % 10 == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, train_loss: {train_loss/num_ol}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, train_loss: {train_loss/num_ol}, val loss: {val_loss}, test loss: {test_loss}\n")
            else:
                print(f"epoch: {i}, train loss: {train_loss}")
                
        