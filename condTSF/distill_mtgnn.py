import torch.nn as nn
from tqdm import tqdm
from model.mtgnn import gtnet
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

        # Define condensed data
        num_total = data['train_loader'].xs.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = self.data['train_loader'].xs[sampled_idx]     
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        
        self.syny = self.data['train_loader'].ys[sampled_idx, :, :, 0]
        self.syny = scaler.transform(self.syny)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
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
        _model = gtnet(True, True, 2, num_nodes,
                  device, predefined_A=None,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,
                  conv_channels=32, residual_channels=32,
                  skip_channels=64, end_channels=128,
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=3e-3)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            _model.train()
            output_syn = _model(synx.transpose(1,3)).squeeze()
            loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():
                    val_loss = _model.test_model(data['val_loader'], scaler, device)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            test_loss = _model.test_model(data['test_loader'], scaler, device)

        return min_i, min_val_loss, test_loss

    
    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny        

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']

        optimizer = torch.optim.Adam([synx, syny], lr=args.lr_feat)
        buffers = [torch.load(args.params + f"replay_buffer_{i}.pt") for i in tqdm(range(args.num_experts))]


        #min_i, val_loss, test_loss = self.test_syn()
        #print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        #with open(args.save_path, 'a') as f:
        #    f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")               
        train_loss_avg = 0
        cnt = 0
        for i in tqdm(range(args.epochs)):
            expert_traj = buffers[np.random.randint(0, len(buffers))]
            student_net = gtnet(True, True, 2, num_nodes,
                  device, predefined_A=None,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,
                  conv_channels=32, residual_channels=32,
                  skip_channels=64, end_channels=128,
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).to(self.device)
            student_net = ReparamModule(student_net)
            student_net.train()
            start_epoch = np.random.randint(0, args.max_start_epoch + 1)

            for multi_step in range(1):                
                start_params = expert_traj[start_epoch]            
                student_params = torch.cat([p.data.clone().to(self.device).reshape(-1) for p in start_params], 0).requires_grad_(True)
                start_params = torch.cat([p.data.clone().to(self.device).reshape(-1) for p in start_params], 0)
    
                target_params = expert_traj[start_epoch + args.expert_epochs]
                target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)            
                param_dist = F.mse_loss(start_params, target_params, reduction="sum")
                
                for step in range(args.syn_steps):
                    output_syn = student_net(synx.transpose(1,3), flat_param=student_params)
                    loss_syn = F.mse_loss(output_syn.squeeze(), syny)
                    gw_syn = torch.autograd.grad(loss_syn, student_params, create_graph=True)[0]
                    student_params = student_params - args.lr_train*gw_syn
    
                param_loss = F.mse_loss(student_params, target_params, reduction="sum")           
                train_loss = param_loss / param_dist
                train_loss_avg += train_loss.item()
                
                optimizer.zero_grad()
                #optimizer_lr.zero_grad()
                train_loss.backward()
    
                optimizer.step()
                #optimizer_lr.step()
    
                #print(f"epoch:{i}, train loss: {grand_loss}, train loss after: {grand_loss_after}")
                #print(f"epoch:{i}, train loss: {train_loss}")
                cnt += 1                      
            
            if (i+1)%10== 0:
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, train loss: {train_loss_avg/cnt}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, train loss: {train_loss_avg/cnt}, val loss: {val_loss}, test loss: {test_loss}\n")
                train_loss_avg = 0
                cnt = 0


# python -m condTSF.distill_mtgnn -de 5 -e 1000 -lrs 0.01 -lrt 0.01 -lrf 0.01 -e 100 -sp results/mtt_mtgnn.txt 
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=3e-3,help='learning rate')
    parser.add_argument('-r', '--reduction_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-lrt', '--lr_train',type=float,default=3e-3,help='learning rate')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-p', '--params', type=str, default='../data/params/METR-LA-mtgnn/')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    parser.add_argument('-ne', '--num_experts', type=int, default=40)
    parser.add_argument('-mse', '--max_start_epoch', type=int, default=7, help='max epoch we can start at')
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