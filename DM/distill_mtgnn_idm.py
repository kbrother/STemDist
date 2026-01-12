from model.mtgnn import gtnet
import torch.nn as nn
from tqdm import tqdm
import torch
import util
import random
import sys
import copy
import argparse
import numpy as np
import torch.optim as optim
from distill_orig import DataDistill
import torch.nn.functional as F


class ImprovedDM(DataDistill):

    def __init__(self, data_dm, data_train, args, device):
        super().__init__(data_dm, args, device)
        self.data_train = data_train
        
        num_nodes = data_dm['train_loader'].xs.shape[2]
        in_dim = data_dm['train_loader'].xs.shape[3]
        seq_len = data_dm['train_loader'].xs.shape[1]
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

        self.model_list = [gtnet(True, True, 2, num_nodes, 
                      device, predefined_A=None, use_static_feat=False,
                      dropout=0.3, subgraph_size=20,
                      node_dim=10, dilation_exponential=1,             
                      seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                      layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).to(device) for _ in range(3)]
        self.opt_list = [torch.optim.Adam(self.model_list[i].parameters(), lr=args.lr_real) for i in range(3)]


    def train(self):
        args = self.args
        data_dm = self.data
        data_train = self.data_train
        synx = self.synx
        device = self.device

        num_nodes = data_dm['train_loader'].xs.shape[2]
        in_dim = data_dm['train_loader'].xs.shape[3]
        seq_len = data_dm['train_loader'].xs.shape[1]
        scaler = data_dm['scaler']        
        out_dim = 1

        min_val_loss = sys.float_info.max
        optimizer = torch.optim.Adam([synx], lr=args.lr_feat)
        for i in tqdm(range(args.epochs)):

            # select model
            curr_idx = random.sample(list(range(len(self.model_list))), 1)[0]
            curr_model = self.model_list[curr_idx]
            curr_opt = self.opt_list[curr_idx]

            data_dm['train_loader'].shuffle()           
            # distirubtion matching
            for it, (x, y) in enumerate((data_dm['train_loader'].get_iterator())): 
                trainx = torch.tensor(x, device=device, dtype=torch.float)                
                trainx = trainx.transpose(1, 3)       
                output_real = curr_model(trainx).squeeze()
                output_real = torch.mean(output_real, dim=0)

                output_syn = curr_model(synx.transpose(1, 3)).squeeze()
                output_syn = torch.mean(output_syn, dim=0)
                _loss = F.mse_loss(output_real, output_syn)

                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()


            self.syny = self.trained_model(synx.transpose(1, 3)).squeeze()
            if (i+1) % args.check == 0:                
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

                
            # Train the current model            
            data_train['train_loader'].shuffle()             
            for it, (x, y) in enumerate(data_train['train_loader'].get_iterator()): 
                if it >= args.syn_step:
                    break

                trainx = torch.tensor(x, device=device, dtype=torch.float)                
                trainx = trainx.transpose(1, 3)
                trainy = torch.tensor(y, device=device, dtype=torch.float)          
                output = curr_model(trainx).squeeze()
                output = scaler.inverse_transform(output)
                curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
                curr_loss /= num_val_entry

                curr_opt.zero_grad()
                curr_loss.backward()
                curr_opt.step()
                
            # Push queue
            self.model_list.append(gtnet(True, True, 2, num_nodes, 
                      device, predefined_A=None, use_static_feat=False,
                      dropout=0.3, subgraph_size=20,
                      node_dim=10, dilation_exponential=1,             
                      seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                      layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).to(device))
            self.opt_list.append(torch.optim.Adam(self.model_list[-1].parameters(), lr=args.lr_real))

            # Pop que
            if len(self.model_list) > args.num_model:
                self.model_list.pop(0)
                self.opt_list.pop(0)
    

# python -m DM.distill_mtgnn_idm -de 0 -d ../data/GBA -e 100 -sp results/idm_gba_1e-2_1e-3 -lrr 1e-2 -lrf 1e-2 -lrs 1e-3 -rr 5e-3 -bd 128 -bt 64 -lp results_param/mtgnn_gba_0.pt -s 0 -ce 200
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-bd', '--batch_size_dm', type=int, default=2**10, help='batch size')
    parser.add_argument('-bt', '--batch_size_train', type=int, default=2**10, help='batch size')
    parser.add_argument('-nm', '--num_model', type=int, default=8, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')
    parser.add_argument('-lrr', '--lr_real',type=float,default=1e-2,help='learning rate')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate')
    parser.add_argument('-rr', '--reduce_rate',type=float,default=2e-2,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-c', '--check',type=int,default=5,help='')
    parser.add_argument('-ce', '--check_epoch',type=int,default=200,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-ss', '--syn_step', type=int, default=20, help='')
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
    dataloader_dm =  util.load_dataset(args.data, args.batch_size_dm)
    dataloader_train =  util.load_dataset(args.data, args.batch_size_train)
    print("load finish")

    algo = ImprovedDM(dataloader_dm, dataloader_train, args, device)    
    algo.train()