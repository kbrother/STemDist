import torch.nn as nn
from tqdm import tqdm
from model import gwnet
import torch
import util
import sys
import copy
import random
from reparam_module import ReparamModule
import torch.nn.functional as F
import numpy as np


class GwaveTraj:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate *  data['train_loader'].xs.shape[0])
        scaler = data['scaler']        

        _shape = [self.num_elems] + list(data['train_loader'].xs.shape[1:])
        self.synx = torch.rand(tuple(_shape), device=device, dtype=torch.float)
        _shape = [self.num_elems] + list(data['train_loader'].ys.shape[1:-1])
        self.syny = torch.rand(_shape, device=device, dtype=torch.float)

        '''
        num_total = data['train_loader'].xs.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = self.data['train_loader'].xs[sampled_idx]     
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        
        self.syny = self.data['train_loader'].ys[sampled_idx, :, :, 0]
        self.syny = scaler.transform(self.syny)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
                '''        
        self.synx = nn.Parameter(self.synx)
        self.syny = nn.Parameter(self.syny)
        
        print(f'feat x shape: {self.synx.shape}')
        print(f'feat y shape: {self.syny.shape}')


    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx.detach().clone()
        syny = self.syny.detach().clone()

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']
        _model = gwnet(self.device, num_nodes, args.dropout, in_dim, args.seq_length, 
                           residual_channels=args.nhid, dilation_channels=args.nhid, 
                           skip_channels=8*args.nhid, end_channels=16*args.nhid)
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=1e-4, weight_decay=0.0001)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(100)):
            _model.train()
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            loss_syn = util.mse(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if i%5 == 0:
                with torch.no_grad():
                    val_loss = _model.test_model(data['val_loader'], scaler)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            test_loss = _model.test_model(data['test_loader'], scaler)

        return min_i, min_val_loss, test_loss

    
    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny        

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']

        syn_lr = nn.Parameter(torch.tensor([args.lr_student]).to(self.device))
        optimizer = torch.optim.Adam([synx, syny], lr=args.learning_rate)
        optimizer_lr = torch.optim.Adam([syn_lr], lr=args.lr_lr)
        buffers = [torch.load(args.params + f"replay_buffer_{i}.pt") for i in range(args.num_experts)]

        '''
        min_i, val_loss, test_loss = self.test_syn()
        print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        with open(args.save_path, 'a') as f:
            f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")               
            '''
        for i in tqdm(range(args.epochs)):
            expert_traj = buffers[np.random.randint(0, len(buffers))]
            student_net = gwnet(self.device, num_nodes, 0, in_dim, args.seq_length, 
                               residual_channels=args.nhid, dilation_channels=args.nhid, 
                               skip_channels=8*args.nhid, end_channels=16*args.nhid).to(self.device)  

            student_net = ReparamModule(student_net)
            start_epoch = np.random.randint(0, args.max_start_epoch)
            start_params = expert_traj[start_epoch]

            student_params = [torch.cat([p.data.to(self.device).reshape(-1) for p in start_params], 0).requires_grad_(True)]
            start_params = torch.cat([p.data.to(self.device).reshape(-1) for p in start_params], 0)

            grand_loss = 0
            check_count = args.syn_steps//args.expert_epoch
            for step in range(args.syn_steps):
                output_syn = student_net(synx.transpose(1, 3), flat_param=student_params[-1]).squeeze()
                loss_syn = util.mse(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - syn_lr*gw_syn)

    
            target_params = expert_traj[start_epoch + args.expert_epoch]
            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)

            param_loss = F.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist = F.mse_loss(start_params, target_params, reduction="sum")
            grand_loss = param_loss / param_dist

            optimizer.zero_grad()
            #optimizer_lr.zero_grad()
            grand_loss.backward()

            optimizer.step()
            #optimizer_lr.step()

            # For checking the reudction of training loss
            '''
            start_params = expert_traj[start_epoch]
            target_params = expert_traj[start_epoch + args.expert_epoch]
            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)
            student_params = [torch.cat([p.data.to(self.device).reshape(-1) for p in start_params], 0).requires_grad_(True)]
            start_params = torch.cat([p.data.to(self.device).reshape(-1) for p in start_params], 0)

            for step in range(args.syn_steps):
                output_syn = student_net(synx.transpose(1, 3), flat_param=student_params[-1]).squeeze()
                loss_syn = util.mse(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, student_params[-1])[0]
                student_params.append(student_params[-1] - syn_lr*gw_syn)

            param_loss = F.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist = F.mse_loss(start_params, target_params, reduction="sum")
            grand_loss_after = param_loss / param_dist
            '''    
            #print(f"epoch:{i}, train loss: {grand_loss}, train loss after: {grand_loss_after}")
            print(f"epoch:{i}, train loss: {grand_loss}")
            if (i+1)%10== 0:
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, train_loss: {grand_loss}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, train_loss: {grand_loss}, val loss: {val_loss}, test loss: {test_loss}\n")