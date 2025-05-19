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


class GradMatch:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_series = int(args.reduce_rate * data['train_loader'].xs.shape[0])
        scaler = data['scaler']
        
        # Define condensed data
        # Define condensed data
        num_series_total = data['train_loader'].xs.shape[0]
        num_nodes_total = data['train_loader'].xs.shape[2]
        
        sampled_idx1 = random.sample(list(range(num_series_total)), self.num_series)
        sampled_idx1.sort()        
        self.synx = self.data['train_loader'].xs[sampled_idx1]     
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        
        self.syny = self.data['train_loader'].ys[sampled_idx1]
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
        seq_len = data['train_loader'].xs.shape[1]
        scaler = data['scaler']
        if len(dataloader['train_loader'].ys.shape) == 4:        
            out_dim = dataloader['train_loader'].ys.shape[3]
        else: 
            out_dim = 1
            
        _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=False,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
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
                    val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():            
            test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))

        return min_i, min_val_loss, test_loss

    
    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        seq_len = data['train_loader'].xs.shape[1]
        scaler = data['scaler']        
        if len(dataloader['train_loader'].ys.shape) == 4:        
            out_dim = dataloader['train_loader'].ys.shape[3]
        else: 
            out_dim = 1
            
        min_val_loss = sys.float_info.max
        optimizer = torch.optim.Adam([synx, syny], lr=args.lr_feat)
        for i in tqdm(range(args.epochs)):
            data['train_loader'].shuffle()
            data['train_loader'].current_ind = 0            
            _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=False,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
            model_params = list(_model.parameters())
            #_model.initialize()
            _model.to(self.device)
            _model.train()
            optimizer_model = torch.optim.Adam(model_params, lr=args.lr_syn)

            grad_loss = 0
            num_ol = 20
            num_real_total = 0
            for ol in range(num_ol):            
                # Compute real gradient           
                
                x, y = data['train_loader'].get_next()                                
                realx = torch.tensor(x, device=self.device, dtype=torch.float)
                realy = torch.tensor(y, device=self.device, dtype=torch.float)
                output_real_temp = _model(realx.transpose(1, 3)).squeeze()
                output_real = scaler.inverse_transform(output_real_temp)
                loss_real, num_real = util.masked_se(output_real, realy, 0.)
                gw_real = torch.autograd.grad(loss_real/num_real, model_params, retain_graph=True)
                gw_real = [_.detach().clone() for _ in gw_real]

                output_syn = _model(synx.transpose(1, 3)).squeeze()
                loss_syn = F.mse_loss(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)
                
                #pbar.close()
                _loss = util.match_loss(gw_syn, gw_real, self.device)                
                # gradient descent                
                grad_loss += _loss.item()
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                if ol == num_ol - 1:
                    break
                    
                num_il = 10
                synx_in, syny_in = synx.detach(), syny.detach()     
                for il in range(num_il):
                    optimizer_model.zero_grad()                    
                    output_syn_in = _model(synx_in.transpose(1,3)).squeeze()
                    loss_syn_in = F.mse_loss(output_syn_in, syny_in)
                    loss_syn_in.backward()
                    optimizer_model.step()

            print(f"epoch: {i}, grad loss: {grad_loss/num_ol}")
            if (i+1) % 10 == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"my epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path + ".txt", 'a') as f:
                    f.write(f"my epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    synx_ = synx.detach().clone().cpu()
                    syny_ = syny.detach().clone().cpu()                    
                    torch.save({'x':synx_, 'y':syny_}, args.save_path + ".pt")
                  


# python -m DC.distill_mtgnn_orig -de 2 -d ../data/METR-LA -e 300 -sp results/dc_metr_la_2e-3 -lrf 1e-2 -lrs 1e-2 -rr 2e-3 -b 256
# python -m DC.distill_mtgnn_orig -de 0 -d ../data/PEMS-BAY -e 300 -sp results/dc_pems_bay_2e-3 -lrf 1e-2 -lrs 1e-3 -rr 2e-3 -b 256
# python -m DC.distill_mtgnn_orig -de 7 -d ../data/AIR-DATA -e 300 -sp results/dc_air_data_2e-3 -lrf 1e-2 -lrs 1e-3 -rr 2e-3 -b 256
# python -m DC.distill_mtgnn_orig -de 0 -d ../data/ELECTRICITY -e 300 -sp results/dc_orig_elec -lrf 1e-2 -lrs 1e-2 -rr 2e-3 -b 256
# python -m DC.distill_mtgnn_orig -de 0 -d ../data/SOLAR -e 300 -sp results/dc_orig_solar_1e-2 -lrf 1e-2 -lrs 1e-2 -rr 2e-3 -b 256
# python -m DC.distill_mtgnn_orig -de 2 -d ../data/TRAFFIC -e 300 -sp results/dc_orig_traffic_1e-3 -lrf 1e-2 -lrs 1e-3 -rr 2e-3 -b 256
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

    algo = GradMatch(dataloader, args, device)    
    algo.train()