import torch
import numpy as np
import argparse
from model.dlinear.dlinear import Model
from model.node_embed import *
import util
from tqdm import tqdm
import random
from model import *
import math
import sys
from CondTSF.reparam_module import ReparamModule
import copy

class Traj_Matching:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_series = int(args.series_reduce_rate * data['train_loader'].xs.shape[0])
        self.num_nodes = int(args.node_reduce_rate * data['train_loader'].xs.shape[2])
        scaler = data['scaler']        

        # Define condensed data
        num_series_total = data['train_loader'].xs_orig.shape[0]
        num_nodes_total = data['train_loader'].xs_orig.shape[2]
        num_seq = data['train_loader'].xs.shape[1]
        num_feat = data['train_loader'].xs.shape[3]

        sampled_idx1 = random.sample(list(range(num_series_total)), self.num_series)
        sampled_idx1.sort()        
        sampled_idx2 = random.sample(list(range(num_nodes_total)), self.num_nodes)
        sampled_idx2.sort()
        
        self.synx = self.data['train_loader'].xs[sampled_idx1][:, :, sampled_idx2, :]     
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)        
        self.syny = self.data['train_loader'].ys[sampled_idx1][:, :, sampled_idx2]
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

        scaler = data['scaler']
        seq_len = data['train_loader'].xs.shape[1]
        _model = Model(seq_len, seq_len, self.num_nodes)
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
        min_val_loss = sys.float_info.max
    
        for i in tqdm(range(200)):
            _model.train()            
            output_syn = _model(synx).squeeze()
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

        optimizer = torch.optim.Adam([synx, syny], lr=args.lr_feat)
        syn_lr = nn.Parameter(torch.FloatTensor(1).to(args.device))
        syn_lr.data = torch.tensor(args.lr_teacher)
        optimizer_lr = torch.optim.Adam([syn_lr], lr=args.lr_lr)
        min_val_loss = sys.float_info.max

        scaler = data['scaler']
        seq_len = data['train_loader'].xs.shape[1]

        for i in tqdm(range(args.epochs)):
            buffer_index = random.randint(0, args.num_experts - 1)
            expert_traj = torch.load(args.params + f"replay_buffer_{buffer_index}.pt")
            data['train_loader'].shuffle()

            student_net = Model(seq_len, seq_len, self.num_nodes)
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
            plugin_params = expert_traj[-1]
            plugin_params = torch.cat([p.data.to(self.device).reshape(-1) for p in plugin_params], 0)
                      
            for _ in range(args.syn_steps):
                output_syn = student_net(synx, flat_param=student_params).squeeze()       
                synx = torch.tensor(synx, device=device, dtype=torch.float) 
                loss_syn = F.mse_loss(output_syn, syny)
                grad = torch.autograd.grad(loss_syn, student_params, create_graph=True, allow_unused=True)[0]
                student_params = student_params - syn_lr * grad

            param_loss = torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
            param_dist = torch.nn.functional.mse_loss(start_params, target_params, reduction="sum")

            param_dist = param_dist / num_params
            param_loss = param_loss / num_params
            grand_loss = param_loss / param_dist

            # CondTSF plugin
            if i % 3 == 0:  
                target_Y = student_net(synx, flat_param=plugin_params).squeeze().detach()
                syny.data = (1 - args.beta) * syny.data + args.beta * target_Y

            else:
                optimizer.zero_grad()
                optimizer_lr.zero_grad()
                grand_loss.backward()
                optimizer.step()
                optimizer_lr.step()
                args.lr_teacher = syn_lr
            
            print(f"epoch:{i}, train loss: {grand_loss.item()}")
            if (i+1)%args.check_freq== 0:
                min_i, val_loss, test_loss = self.test_syn()
                val_loss = math.sqrt(val_loss)
                test_loss = math.sqrt(test_loss)
                print(f"epoch: {i}, min i: {min_i}, train_loss: {grand_loss.item()}, val loss: {val_loss}, test loss: {test_loss}")
                
                with open(args.save_path + ".txt", 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, train_loss: {grand_loss.item()}, val loss: {val_loss}, test loss: {test_loss}\n")
                
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    synx_ = synx.detach().clone().cpu()
                    syny_ = syny.detach().clone().cpu()                    
                    torch.save({'x':synx_, 'y':syny_}, args.save_path + ".pt")


# python -m CondTSF.distill_dlinear_condtsf -e 100 -d '../data/AURORA_final' --params '../data/params/AURORA-Dlinear/' -sr 0.005 -lrf 1e-3 -lrs 1e-2 -sp results/condtsf_aurora_0 -de 0 -s 0
# python -m CondTSF.distill_dlinear_condtsf -e 300 -d '../data/GBA' --params '../data/params/GBA-Dlinear/' -sr 0.005 -lrf 1e-3 -lrs 1e-2 -sp results/condtsf_aurora_3_2 -de 5 -s 0

if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch sizefor real data')#128
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-3,help='learning rate for testing on synthetic data')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate for updating synthetic data')
    parser.add_argument('-sr', '--series_reduce_rate',type=float,default=2e-2,help='series reduce rate')
    parser.add_argument('-nr', '--node_reduce_rate',type=float,default=2e-2,help='node reduce rate')
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
    parser.add_argument('--beta', type=float, default=0.01, help='CondTSF addtive ratio')
    parser.add_argument('--check_freq', type=int, default=10, help='how often to check the synthetic data')

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


# python -m CondTSF.distill_dlinear_condtsf -e 100 -d '../data/GBA' --params '../data/params/GBA-Dlinear/' -sr 0.1 -nr 0.1 -lrf 1e-3 -lrs 1e-2 -sp results/condtsf_aurora_3_2 -de 5 -s 0
