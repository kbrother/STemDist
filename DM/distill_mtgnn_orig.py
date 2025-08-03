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
from distill_orig import DataDistill


class DistMatch(DataDistill):

    def __init__(self, data, args, device):
        super().__init__(data, args, device)

        num_nodes = dataloader['train_loader'].xs.shape[2]
        in_dim = dataloader['train_loader'].xs.shape[3]
        seq_len = dataloader['train_loader'].xs.shape[1]
        out_dim = 1
        self.trained_model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=False,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
        state_dict = torch.load(args.load_path, map_location=device)
        self.trained_model.load_state_dict(state_dict)
        self.trained_model.to(device)
        
        
    def train(self):
        args = self.args
        data = self.data
        synx = self.synx
        device = self.device

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        seq_len = data['train_loader'].xs.shape[1]
        scaler = data['scaler']        
        out_dim = 1
        
        min_val_loss = sys.float_info.max
        optimizer = torch.optim.Adam([synx], lr=args.lr_feat)
        for i in tqdm(range(args.epochs)):        
            data['train_loader'].shuffle()           
            for it, (x, y) in enumerate((data['train_loader'].get_iterator())): 
                _model = gtnet(True, True, 2, num_nodes, 
                      device, predefined_A=None, use_static_feat=False,
                      dropout=0.3, subgraph_size=20,
                      node_dim=10, dilation_exponential=1,             
                      seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                      layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
                _model = _model.to(device)
                    
                trainx = torch.tensor(x, device=device, dtype=torch.float)                
                trainx = trainx.transpose(1, 3)       
                output_real = _model(trainx).squeeze()
                output_real = torch.mean(output_real, dim=0)

                output_syn = _model(synx.transpose(1, 3)).squeeze()
                output_syn = torch.mean(output_syn, dim=0)
                _loss = F.mse_loss(output_real, output_syn)

                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

            self.syny = self.trained_model(synx.transpose(1, 3)).squeeze()
            if (i+1) % args.check_freq == 0:                
                val_sum, test_sum = 0, 0
                num_iter = 3
                for j in range(num_iter):
                    min_i, val_loss, test_loss = self.test_syn()
                    val_sum += val_loss
                    test_sum += test_loss
                    print(f"my epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path + ".txt", 'a') as f:
                    val_avg = val_sum/num_iter
                    f.write(f"my epoch: {i}, val loss: {val_avg}, test loss: {test_sum/num_iter}\n")
                
                if min_val_loss > val_avg:
                    min_val_loss = val_avg
                    synx_ = self.synx.detach().clone().cpu()
                    syny_ = self.syny.detach().clone().cpu()                    
                    torch.save({'x':synx_, 'y':syny_}, args.save_path + ".pt")


# python -m DM.distill_mtgnn_orig -de 0 -d ../data/GBA -e 100 -sp results/dm_gba_1e-2_1e-3 -lrf 1e-2 -lrs 1e-3 -rr 5e-3 -b 128 -lp results/gba_0.pt
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**10, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate')
    parser.add_argument('-rr', '--reduce_rate',type=float,default=2e-2,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-c', '--check',type=int,default=5,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    parser.add_argument('-lp', '--load_path', type=str, default='results/')
    
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    #device = torch.device(f"cpu")
    dataloader =  util.load_dataset(args.data, args.batch_size)
    print("load finish")

    algo = DistMatch(dataloader, args, device)    
    algo.train()