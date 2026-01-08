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


class GradMatchCluster:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_series = round(args.series_reduce_rate * data['train_loader'].xs.shape[0])
        self.num_nodes = data['train_loader'].xs.shape[2]
        scaler = data['scaler']
        
        # Define condensed data
        # Define condensed data
        num_series_total = data['train_loader'].xs_orig.shape[0]
        num_nodes_total = data['train_loader'].xs_orig.shape[2]
        num_seq = data['train_loader'].xs.shape[1]
        num_feat = data['train_loader'].xs.shape[3]
                
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
        
        val_sum, test_sum = 0, 0
        num_iter = 3
        for i in range(num_iter):
            min_i, val_loss, test_loss = self.test_syn()
            print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
            val_sum += val_loss
            test_sum += test_loss
        with open(args.save_path + ".txt", 'a') as f:
            f.write(f"initial, min i: {min_i}, val loss: {val_sum/num_iter}, test loss: {test_sum/num_iter}\n")
        
    
    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx.detach().clone()
        syny = self.syny.detach().clone()
        
        num_nodes = data['train_loader'].xs_orig.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        seq_len = data['train_loader'].xs.shape[1]
        scaler = data['scaler']
        if len(dataloader['train_loader'].ys.shape) == 4:        
            out_dim = dataloader['train_loader'].ys.shape[3]
        else: 
            out_dim = 1
            
        _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim)   
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
        min_val_loss = sys.float_info.max      

        nm_input_real = np.mean(data['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
        nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
        nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
        nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim
         
        nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
        nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
        nm_input_syn = torch.reshape(nm_input_syn, (self.num_nodes, -1))

        _weight = torch.tensor(data['train_loader'].label_cnt, device=device)
        _weight = _weight.unsqueeze(0).unsqueeze(0)
        num_sample = round(self.num_nodes/4)
        for i in tqdm(range(args.check_epoch)):  
            _model.train()
            _order = list(range(self.num_nodes))
            random.shuffle(_order)
            for j in range(4):                
                if j < 3:
                    _idx = _order[num_sample*j:num_sample*(j+1)]
                else:
                    _idx = _order[num_sample*j:]                                
                    
                curr_weight = _weight[:,:,_idx] / torch.sum(_weight[:,:,_idx]) * num_sample
                _model.embed_forward(nm_input_syn[_idx])
                output_syn = _model(synx[:,:,_idx,:].transpose(1,3)).squeeze()
                #loss_syn = F.mse_loss(output_syn, syny)                
                loss_syn = torch.square(output_syn - syny[:,:,_idx]) * curr_weight
                loss_syn = torch.mean(loss_syn)
                optimizer.zero_grad()
                loss_syn.backward()
                optimizer.step()

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():
                    _model.embed_forward(nm_input_real)
                    val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

                print(val_loss)
        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            _model.embed_forward(nm_input_real)
            test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))

        return min_i, min_val_loss, test_loss

    
    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny
        _ratio = 3

        in_dim = data['train_loader'].xs.shape[3]
        seq_len = data['train_loader'].xs.shape[1]
        scaler = data['scaler']        
        if len(dataloader['train_loader'].ys.shape) == 4:        
            out_dim = dataloader['train_loader'].ys.shape[3]
        else: 
            out_dim = 1
            
        min_val_loss = sys.float_info.max
        optimizer = torch.optim.Adam([synx, syny], lr=args.lr_feat)

        nm_input_real = np.mean(data['train_loader'].xs, axis=0)  # seq_length x num nodes x in_dim
        nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
        nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
        nm_input_real = torch.reshape(nm_input_real, (self.num_nodes, -1))   # num_nodes x seq length*in_dim

        _weight = torch.tensor(data['train_loader'].label_cnt, device=device)
        _weight = _weight.unsqueeze(0).unsqueeze(0)
        num_sample = round(self.num_nodes/4)
        for i in tqdm(range(args.epochs)):
            data['train_loader'].shuffle()
            data['train_loader'].current_ind = 0            
            _model = gtnet(True, True, 2, self.num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim)   
            model_params = list(_model.parameters())
            #_model.initialize()
            _model.to(self.device)
            _model.train()
            optimizer_model = torch.optim.Adam(model_params, lr=args.lr_syn)

            grad_loss = 0
            num_ol = 20
            num_real_total = 0
            for ol in range(num_ol):         

                _order = list(range(self.num_nodes))
                random.shuffle(_order)                
                x, y = data['train_loader'].get_next()                                
                realx = torch.tensor(x, device=self.device, dtype=torch.float)
                realy = torch.tensor(y, device=self.device, dtype=torch.float)  
                for ii in range(4):
                    if ii < 3:
                        _idx = _order[num_sample*ii:num_sample*(ii+1)]
                    else:
                        _idx = _order[num_sample*ii:]                                                      
                    curr_weight = _weight[:,:,_idx] / torch.sum(_weight[:,:,_idx]) * num_sample
                    _model.embed_forward(nm_input_real[_idx])                    
                    
                    output_real_temp = _model(realx[:,:,_idx,:].transpose(1,3)).squeeze()                
                    output_real = scaler.inverse_transform(output_real_temp)
    
                    loss_real = torch.square(output_real - realy[:,:,_idx]) * curr_weight
                    loss_real = torch.mean(loss_real)                
                    gw_real = torch.autograd.grad(loss_real, model_params, retain_graph=True)
                    gw_real = [_.detach().clone() for _ in gw_real]
    
                    nm_input_syn = torch.mean(synx[:,:,_idx,:], dim=0) # seq_length x num_nodes x in_dim
                    nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
                    nm_input_syn = torch.reshape(nm_input_syn, (nm_input_syn.shape[0], -1))           
                    _model.embed_forward(nm_input_syn)
    
                    output_syn = _model(synx[:,:,_idx,:].transpose(1,3)).squeeze()
                    loss_syn = torch.square(output_syn - syny[:,:,_idx]) * curr_weight
                    loss_syn = torch.mean(loss_syn)
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
                nm_input_syn_in = torch.mean(synx_in, dim=0)   # seq_length x num_nodes x in_dim
                nm_input_syn_in = torch.transpose(nm_input_syn_in, 0, 1)  # num nodex x seq_length x in_dim
                nm_input_syn_in = torch.reshape(nm_input_syn_in, (self.num_nodes, -1))
                for il in range(num_il):
                    _order = list(range(self.num_nodes))
                    random.shuffle(_order)        
                    for ii in range(4):
                        if ii < 3:
                            _idx = _order[num_sample*ii:num_sample*(ii+1)]
                        else:
                            _idx = _order[num_sample*ii:]       
                        curr_weight = _weight[:,:,_idx] / torch.sum(_weight[:,:,_idx]) * num_sample
                        _model.embed_forward(nm_input_syn_in[_idx])                    
                        output_syn_in = _model(synx_in[:,:,_idx,:].transpose(1,3)).squeeze()
                        #loss_syn_in = F.mse_loss(output_syn_in, syny_in)
                        loss_syn_in = torch.square(output_syn_in - syny_in[:,:,_idx]) * curr_weight
                        loss_syn_in = torch.mean(loss_syn_in)
                        optimizer_model.zero_grad()
                        loss_syn_in.backward()
                        optimizer_model.step()

            print(f"epoch: {i}, grad loss: {grad_loss/num_ol}")
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
                    synx_ = synx.detach().clone().cpu()
                    syny_ = syny.detach().clone().cpu()                    
                    torch.save({'x':synx_, 'y':syny_ ,'w': _weight}, args.save_path + ".pt")
                  

# python -m stemdist -de 2 -d ../data/GBA -e 100 -sp results/dc_dsa_cluster_gba_1e-3_1e-3 -lrf 1e-3 -lrs 1e-3 -srr 0.1 -nrr 0.1 -b 256 -ned 32 -s 1 -c 5
# python -m stemdist -de 3 -d ../data/GLA -e 100 -sp results/dc_dsa_cluster_gla_1e-3_1e-3 -lrf 1e-3 -lrs 1e-3 -srr 0.1 -nrr 0.1 -b 128 -ned 32 -s 2 -c 5
# python -m stemdist -de 1 -d ../data/ERA5 -e 100 -sp results/dc_dsa_cluster_era5_1e-2 -lrf 1e-2 -lrs 1e-2 -srr 0.1 -nrr 0.1 -b 64 -ned 32 -s 0 -c 5
# python -m stemdist -de 1 -d ../data/AURORA -e 100 -sp results/dc_dsa_cluster_aurora_1e-2_1e-3 -lrf 1e-2 -lrs 1e-3 -srr 0.1 -nrr 0.1 -b 64 -ned 32 -s 1 -c 5
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**10, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate')
    parser.add_argument('-nrr', '--node_reduce_rate',type=float,default=1,help='learning rate')
    parser.add_argument('-srr', '--series_reduce_rate',type=float,default=2e-2,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-ned', '--ne_dim',type=int,default=32,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    parser.add_argument('-c', '--check_freq', type=int, default=10, help='')
    parser.add_argument('-ce', '--check_epoch', type=int, default=10, help='')
    parser.add_argument('-lt', '--loader_type', type=str, default='kmeans', help='data path')
    parser.add_argument('-nr', '--num_rows', type=int, default=10, help='')
    parser.add_argument('-nc', '--num_cols', type=int, default=10, help='')
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    #device = torch.device(f"cpu")
    dataloader =  util.load_dataset(args.data, args.batch_size, args.loader_type, args.node_reduce_rate, device,
                                   args.num_rows, args.num_cols)
    print("load finish")

    algo = GradMatchCluster(dataloader, args, device)    
    algo.train()
