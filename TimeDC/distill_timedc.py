import torch
import numpy as np
import argparse
# from model.dlinear.dlinear import Model
# from model.node_embed import *
import util
from tqdm import tqdm
import random
from model import *
import math
import sys
from TimeDC.reparam_module import ReparamModule
import copy
from TimeDC.network_patch import TSFE_Model
import torch.nn as nn
import torch.nn.functional as F
import time

class Traj_Matching:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_series = int(args.series_reduce_rate * data['train_loader'].xs.shape[0])
        scaler = data['scaler']        

        # Define condensed data
        num_series_total = data['train_loader'].xs.shape[0]
        sampled_idx1 = random.sample(list(range(num_series_total)), self.num_series)
        
        self.synx = self.data['train_loader'].xs[sampled_idx1, :, :, 0]
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)        
        self.syny = self.data['train_loader'].ys[sampled_idx1, :, :]
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
        num_features = data['train_loader'].xs.shape[3]
        scaler = data['scaler']
        num_nodes = data['train_loader'].xs.shape[2]
        seq_len = data['train_loader'].xs.shape[1]
        # _model = Model(seq_len, seq_len, num_nodes)
        _model = TSFE_Model(args, num_features=num_features).to(args.device)
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_teacher)
        min_val_loss = sys.float_info.max
    
        for i in tqdm(range(200)):
            _model.train()            
            # synx = synx.squeeze(3)
            output_syn,_, _ = _model(synx)
            loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%args.check_freq == 0:
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

        optimizer = torch.optim.SGD([synx, syny], lr=args.lr_feat, momentum=0.5) #torch.optim.Adam([synx, syny], lr=args.lr_feat)
        # syn_lr = nn.Parameter(torch.FloatTensor(1).to(args.device))
        syn_lr = args.lr_teacher
        # optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
        min_val_loss = sys.float_info.max

        scaler = data['scaler']
        num_nodes = data['train_loader'].xs.shape[2]
        num_features = data['train_loader'].xs.shape[3]
        seq_len = data['train_loader'].xs.shape[1]

        for i in range(10):
            start_time = time.time()
            buffer_index = random.randint(0, args.num_experts - 1)
            expert_traj = torch.load("../data/params/AURORA-TimeDC/AURORA_TimeDC_replay_buffer_0.pt")
            data['train_loader'].shuffle()

            # student_net = Model(seq_len, seq_len, num_nodes)
            student_net = TSFE_Model(args, num_features=num_features).to(args.device)
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
                # synx 데이터를 하나씩 처리하고, 각 샘플에서 노드를 4 그룹으로 나눠서 처리
                total_loss_syn = 0
                
                for j in range(synx.shape[0]):  # 시계열 샘플별로 반복
                    single_synx = synx[j:j+1]  # 하나의 시계열 샘플 선택 [1, seq_len, num_nodes]
                    single_syny = syny[j:j+1]  # 하나의 타겟 선택 [1, seq_len, num_nodes]
                    
                    num_nodes = single_synx.shape[2]
                    num_sample = num_nodes // args.num_groups  # 각 그룹당 노드 수
                    node_idx = random.sample(range(num_nodes), num_sample)
                                            
                    # 노드 그룹 선택
                    group_synx = single_synx[:, :, node_idx]  # [1, seq_len, group_size]
                    group_syny = single_syny[:, :, node_idx]  # [1, seq_len, group_size]
                    
                    output_syn, _, _ = student_net(group_synx, flat_param=student_params)       
                    # print(output_syn.shape)
                    loss_syn = F.mse_loss(output_syn, group_syny)
                    total_loss_syn += loss_syn
                
                # 전체 배치에 대한 평균 손실로 그래디언트 계산
                total_samples = synx.shape[0] * 4  # 각 샘플당 4개 그룹
                avg_loss_syn = total_loss_syn / total_samples
                grad = torch.autograd.grad(avg_loss_syn, student_params, create_graph=True, allow_unused=True)[0]
                student_params = student_params - syn_lr * grad

            param_loss = torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
            param_dist = torch.nn.functional.mse_loss(start_params, target_params, reduction="sum")
            # print(param_loss, param_dist)

            param_dist = param_dist / num_params
            param_loss = param_loss / num_params
            grand_loss = param_loss / param_dist


            optimizer.zero_grad()
            # optimizer_lr.zero_grad()
            grand_loss.backward()
            optimizer.step()
            # optimizer_lr.step()
            
            if i >= 5:
                print(f"epoch: {i}, train loss: {grand_loss.item()}, time: {time.time() - start_time}")

            # print(f"epoch:{i}, train loss: {grand_loss.item()}")
            # if (i+1)%args.check_freq== 0:
            #     min_i, val_loss, test_loss = self.test_syn()
            #     val_loss = math.sqrt(val_loss)
            #     test_loss = math.sqrt(test_loss)
            #     print(f"epoch: {i}, min i: {min_i}, train_loss: {grand_loss.item()}, val loss: {val_loss}, test loss: {test_loss}")
                
            #     with open(args.save_path + ".txt", 'a') as f:
            #         f.write(f"epoch: {i}, min i: {min_i}, train_loss: {grand_loss.item()}, val loss: {val_loss}, test loss: {test_loss}\n")
                
            #     if val_loss < min_val_loss:
            #         min_val_loss = val_loss
            #         synx_ = synx.detach().clone().cpu()
            #         syny_ = syny.detach().clone().cpu()                    
            #         torch.save({'x':synx_, 'y':syny_}, args.save_path + ".pt")


# python -m TimeDC.distill_mtgnn_timedc -de 0 -e 100 -d '../data/GBA' --params '../data/params/GBA-MTGNN/' -sp results/mtt_gba.txt -lrf 1e-2 -lrs 1e-3 -rr 0.005 

# python -m TimeDC.distill_mtgnn_timedc -de 4 -e 100 -d '../data/GBA' --params '../data/params/GBA-MTGNN/' -sp results/mtt_gba.txt -lrf 1e-2 -lrs 1e-3 -rr 0.005 


if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch sizefor real data')#128
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-3,help='learning rate for testing on synthetic data')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate for updating synthetic data')
    parser.add_argument('-rr', '--series_reduce_rate',type=float,default=2e-2,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/') 
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    parser.add_argument('-ng', '--num_groups', type=int, default=4, help='')
    parser.add_argument('-cf', '--check_freq', type=int, default=10, help='')
    parser.add_argument('--lr_teacher', type=float, default=5e-4, help='initialization for student params learning rate')
    # parser.add_argument('--lr_lr', type=float, default=1e-7, help='learning rate for updating... learning rate')
    parser.add_argument('--num_experts', type=int, default=20, help='')
    parser.add_argument('--params', type=str, default='../data/params/METR-LA-MTGNN/')
    parser.add_argument('--max_start_epoch', type=int, default=20, help='max epoch we can start at')
    parser.add_argument('--expert_epoch', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--beta', type=float, default=0.01, help='CondTSF addtive ratio')

    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=1, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')


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