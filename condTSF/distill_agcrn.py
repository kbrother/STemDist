import torch.nn as nn
from tqdm import tqdm
from model.agcrn import AGCRN
import torch
import util
import sys
import copy
import random
from reparam_module import ReparamModule
import torch.nn.functional as F
import numpy as np
import argparse


class condTSC:

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
        _model = AGCRN(args, num_nodes, in_dim)
        _model.to(self.device)
        for p in _model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
                
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(100)):
            _model.train()
            output_syn = _model(synx).squeeze()
            loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():
                    val_loss = _model.test_model(data['val_loader'], scaler, self.device)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            test_loss = _model.test_model(data['test_loader'], scaler, self.device)

        return min_i, min_val_loss, test_loss

    
    def train(self):
        args = self.args
        data = self.data    

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']

        optimizer = torch.optim.Adam([self.synx, self.syny], lr=args.lr_feat)
        buffers = [torch.load(args.params + f"replay_buffer_{i}.pt") for i in tqdm(range(args.num_experts))]

    
        min_i, val_loss, test_loss = self.test_syn()
        print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        with open(args.save_path, 'a') as f:
            f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")               
        for i in tqdm(range(args.epochs)):
            expert_traj = buffers[np.random.randint(0, len(buffers))]
            student_net = AGCRN(args, num_nodes, in_dim)
            student_net = student_net.to(self.device)    
            student_net = ReparamModule(student_net)
            
            start_epoch = np.random.randint(0, args.max_start_epoch)
            start_params = expert_traj[start_epoch]            
            student_params = torch.cat([p.data.clone().to(self.device).reshape(-1) for p in start_params], 0).requires_grad_(True)
            start_params = torch.cat([p.data.clone().to(self.device).reshape(-1) for p in start_params], 0)

            target_params = expert_traj[start_epoch + args.expert_epochs]
            target_params = torch.cat([p.data.clone().to(self.device).reshape(-1) for p in target_params], 0)            
            param_dist = F.mse_loss(start_params, target_params, reduction="sum")
            
            for step in range(args.syn_steps):
                perm = torch.randperm(self.synx.shape[0])[:args.batch_syn]
                synx = self.synx[perm]
                syny = self.syny[perm]
                
                output_syn = student_net(synx, flat_param=student_params).squeeze()
                loss_syn = F.mse_loss(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, student_params, create_graph=True)[0]
                student_params = student_params - args.lr_train*gw_syn

            param_loss = F.mse_loss(student_params, target_params, reduction="sum")           
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
                print(f"epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")


# python -m condTSC.distill_agcrn -de 2 -e 100 -sp results/metr-la.txt -lrf 1
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-lrt', '--lr_train',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate')
    parser.add_argument('-bs', '--batch_syn', type=int, default=2**3, help='batch size')
    parser.add_argument('-rr', '--reduction_rate',type=float,default=1e-3,help='learning rate')    
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')

    parser.add_argument('-r', '--rnn_units', type=int, default=2**6, help='rnn hidden unit')
    parser.add_argument('-nl', '--num_layers', default=2, type=int)
    parser.add_argument('-ed', '--embed_dim', default=10, type=int)
    
    parser.add_argument('-p', '--params', type=str, default='../data/params/METR-LA/')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    parser.add_argument('-ne', '--num_experts', type=int, default=40)
    parser.add_argument('-mse', '--max_start_epoch', type=int, default=4, help='max epoch we can start at')
    parser.add_argument('-ss', '--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('-ee', '--expert_epochs', type=int, default=2, help='how many expert epochs the target params are')
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")

    algo = condTSC(dataloader, args, device)
    algo.train()