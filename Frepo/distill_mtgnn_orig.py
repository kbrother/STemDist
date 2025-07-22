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


class Frepo(DataDistill):

    def train(self):
        args, data = self.args, self.data
        device= self.device

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        seq_len = data['train_loader'].xs.shape[1]
        scaler = data['scaler']        
        out_dim = 1

        min_val_loss = sys.float_info.max
        optimizer_feat = torch.optim.Adam([self.synx, self.syny], lr=args.lr_feat)

        num_model = 3
        model_list = [gtnet(True, True, 2, num_nodes, 
                      device, predefined_A=None, use_static_feat=False,
                      dropout=0.3, subgraph_size=20,
                      node_dim=10, dilation_exponential=1,             
                      seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                      layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).to(device) for _ in range(num_model)]

        optimizer_list = [torch.optim.Adam(model_list[i].parameters(), lr=args.lr_syn) for i in range(num_model)]
        update_cnt = [0 for _ in range(num_model)]
        
        for i in tqdm(range(args.epochs)):        
            train_loss = 0
            num_loop = 0

            data['train_loader'].shuffle()
            data['train_loader'].current_ind = 0    
            while num_loop < 10:
                x, y = data['train_loader'].get_next() 
                curr_idx = random.randint(0, num_model-1)
                curr_model = model_list[curr_idx]
                curr_optimizer = optimizer_list[curr_idx]
                
                curr_model.eval()
                realx = torch.tensor(x, device=device, dtype=torch.float)                
                realx = realx.transpose(1, 3)
                realy = torch.tensor(y, device=device, dtype=torch.float)          
                output_real = curr_model(realx).squeeze()  # batch x 12 x num nodes
                output_real = output_real.reshape(output_real.shape[0], -1)  # batch x 12*num_nodes

                output_syn = curr_model(self.synx.transpose(1, 3)).squeeze()  # batch x 12 x num nodes
                output_syn = output_syn.reshape(output_syn.shape[0], -1)  # batch x 12*num_nodes

                K_ts = torch.mm(output_real, output_syn.T)
                K_ss = torch.mm(output_syn, output_syn.T) + torch.eye(output_syn.shape[0]).to(self.device)

                syny = self.syny.reshape(self.syny.shape[0], -1)
                realy = realy.reshape(realy.shape[0], -1)                
                _loss = realy - torch.mm(K_ts, torch.mm(torch.inverse(K_ss), syny))
                _loss = torch.mean(torch.norm(_loss, dim=1))

                optimizer_feat.zero_grad()
                _loss.backward()
                optimizer_feat.step()
                train_loss += _loss.item()
                num_loop += 1
                
                curr_model.train()
                synx_in, syny_in = self.synx.detach(), self.syny.detach()
                curr_optimizer.zero_grad()
                output_syn_in = curr_model(synx_in.transpose(1,3)).squeeze()
                loss_syn_in = F.mse_loss(output_syn_in, syny_in)
                loss_syn_in.backward()
                curr_optimizer.step()

                update_cnt[curr_idx] += 1
                #print(update_cnt[curr_idx])
                if update_cnt[curr_idx] >= 10:
                    model_list[curr_idx] = gtnet(True, True, 2, num_nodes, 
                                              device, predefined_A=None, use_static_feat=False,
                                              dropout=0.3, subgraph_size=20,
                                              node_dim=10, dilation_exponential=1,             
                                              seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                                              layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).to(device) 
                    optimizer_list[curr_idx] = torch.optim.Adam(model_list[curr_idx].parameters(), lr=args.lr_syn)
                    update_cnt[curr_idx] = 0
                
                if data['train_loader'].current_ind == 0:
                    data['train_loader'].shuffle()
                    
            print(f"epoch: {i}, train loss: {train_loss/num_loop}")
            if (i+1) % args.check == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"my epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path + ".txt", 'a') as f:
                    f.write(f"my epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    synx_ = self.synx.detach().clone().cpu()
                    syny_ = self.syny.detach().clone().cpu()                    
                    torch.save({'x':synx_, 'y':syny_}, args.save_path + ".pt")


# python -m Frepo.distill_mtgnn_orig -de 2 -d ../data/GBA_24 -e 100 -sp results/frepo_gba -lrf 1e-3 -lrs 1e-3 -rr 1e-2 -b 128 -c 1 
# python -m Frepo.distill_mtgnn_orig -de 3 -d ../data/GLA_24 -e 100 -sp results/frepo_gla -lrf 1e-3 -lrs 1e-3 -rr 1e-2 -b 128 -c 1 
# python -m Frepo.distill_mtgnn_orig -de 4 -d ../data/ERA5_24 -e 100 -sp results/frepo_era5 -lrf 1e-3 -lrs 1e-3 -rr 1e-2 -b 128 -c 5 
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
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    parser.add_argument('-c', '--check',type=int,default=5,help='')
    
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

    algo = Frepo(dataloader, args, device)    
    algo.train()