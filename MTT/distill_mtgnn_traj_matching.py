import torch
import numpy as np
import argparse
from model.mtgnn import gtnet
from model.node_embed import *
import util
from tqdm import tqdm
import random
from model import *
import torch.optim as optim
import math
import sys
from reparam_module import ReparamModule
import copy
from distill_orig import DataDistill

class Traj_Matching(DataDistill):

    def __init__(self, data, args, device):
        super().__init__(data, args, device)
            

    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny        

        optimizer = torch.optim.Adam([synx, syny], lr=args.lr_feat)
        syn_lr = nn.Parameter(torch.FloatTensor(1).to(args.device))
        syn_lr.data = torch.tensor(args.lr_teacher)
        optimizer_lr = torch.optim.Adam([syn_lr], lr=args.lr_lr)

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        if len(data['train_loader'].ys.shape) == 4:        
            out_dim = data['train_loader'].ys.shape[3]
        else: 
            out_dim = 1
        
        for i in tqdm(range(args.epochs)):
            buffer_index = random.randint(0, args.num_experts - 1)
            expert_traj = torch.load(args.params + f"replay_buffer_{buffer_index}.pt")
            data['train_loader'].shuffle()

            student_net = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=False,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=args.seq_length, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim) 
            student_net.to(self.device)

            student_net = ReparamModule(student_net)
            student_net.train()
            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

            start_epoch = np.random.randint(0, args.max_start_epoch)
            starting_params = expert_traj[start_epoch]

            target_params = expert_traj[start_epoch + args.expert_epoch]
            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)
            student_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)
            start_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            for _ in range(args.syn_steps):                      
                output_syn = student_net(synx.transpose(1, 3), flat_param=student_params).squeeze()
                loss_syn = F.mse_loss(output_syn, syny)
                grad = torch.autograd.grad(loss_syn, student_params, create_graph=True, allow_unused=True)[0]

                student_params = student_params - syn_lr * grad

            param_loss = torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
            param_dist = torch.nn.functional.mse_loss(start_params, target_params, reduction="sum")

            param_dist = param_dist / num_params
            param_loss = param_loss / num_params
            grand_loss = param_loss / param_dist


            optimizer.zero_grad()
            optimizer_lr.zero_grad()
            grand_loss.backward()
            optimizer.step()
            optimizer_lr.step()

            args.lr_teacher = syn_lr
            
            print(f"epoch:{i}, train loss: {grand_loss.item()}")
            if (i+1)%10== 0:
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, train_loss: {grand_loss.item()}, val loss: {math.sqrt(val_loss)}, test loss: {math.sqrt(test_loss)}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, train_loss: {grand_loss.item()}, val loss: {math.sqrt(val_loss)}, test loss: {math.sqrt(test_loss)}\n")

# python -m MTT.traj_matching_mtgnn -de 2 -e 1000 -d '../data/METR-LA' --params '../data/params/METR-LA-MTGNN/' -sp results/tm_metr.txt -lrf 1e-2 -lrs 1e-3 -sr 2e-3 
# python -m MTT.traj_matching_mtgnn -de 4 -e 1000 -d '../data/AURORA' --params '../data/params/AURORA-MTGNN/' -sp results/tm_aurora.txt -lrf 1e-2 -lrs 1e-3 -sr 2e-3 
# python -m MTT.traj_matching_mtgnn -de 1 -e 1000 -d '../data/PEMS-BAY' --params '../data/params/PEMS-BAY-MTGNN/' -sp results/tm_pems.txt -lrf 1e-2 -lrs 1e-3 -sr 2e-3 


if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch sizefor real data')#128
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-3,help='learning rate for testing on synthetic data')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate for updating synthetic data')
    parser.add_argument('-sr', '--series_reduce_rate',type=float,default=2e-2,help='learning rate')
    parser.add_argument('-nr', '--node_reduce_rate',type=float,default=1e-1,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/') 
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-dr', '--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    
    parser.add_argument('--lr_teacher', type=float, default=5e-4, help='initialization for student params learning rate')
    parser.add_argument('--lr_lr', type=float, default=1e-6, help='learning rate for updating... learning rate')
    parser.add_argument('--num_experts', type=int, default=20, help='')
    parser.add_argument('--params', type=str, default='../data/params/METR-LA-MTGNN/')
    parser.add_argument('--max_start_epoch', type=int, default=4, help='max epoch we can start at')
    parser.add_argument('--expert_epoch', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=10, help='how many steps to take on synthetic data')
    parser.add_argument('--ne_dim', type=int, default=32, help='')
    
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader =  util.load_dataset(args.data, 128)
    print("load finish")

    algo = Traj_Matching(dataloader, args, device)    
    algo.train()