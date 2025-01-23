import torch.nn as nn
from tqdm import tqdm
from model.grnn import GRNN
import torch
import util
import sys
import copy
import random
import argparse
import numpy as np
import torch.nn.functional as F


class GradMatch:

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
        _model = GRNN(args, num_nodes, in_dim)
        for p in _model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=3e-3)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(500)):
            _model.train()
            output_syn = _model(synx)
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
        synx, syny = self.synx, self.syny

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']

        #min_i, val_loss, test_loss = self.test_syn()
        #print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        #with open(args.save_path, 'a') as f:
        #    f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        
        optimizer = torch.optim.Adam([synx, syny], lr=args.lr_feat)
        for i in tqdm(range(args.epochs)):
            data['train_loader'].shuffle()
            data['train_loader'].current_ind = 0            
            _model = GRNN(args, num_nodes, in_dim)
            for p in _model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)            
            model_params = list(_model.parameters())
            #_model.initialize()
            _model.to(self.device)
            _model.train()
            optimizer_model = torch.optim.Adam(model_params, lr=args.lr_syn)

            grad_loss = 0
            num_ol = 20
            num_real_total = 0
            for ol in range(num_ol):
                output_syn = _model(synx)
                loss_syn = F.mse_loss(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

                # Compute real gradient                            
                x, y = data['train_loader'].get_next()
                realx = torch.tensor(x, device=self.device, dtype=torch.float)
                realy = torch.tensor(y, device=self.device, dtype=torch.float)
                realy = realy[:,:,:,0]  # batch x seq len x num node
                output_real_temp = _model(realx)
                output_real = scaler.inverse_transform(output_real_temp)
                loss_real, num_real = util.masked_se(output_real, realy, 0.)
                gw_real = torch.autograd.grad(loss_real/num_real, model_params, retain_graph=True)
                gw_real = list((_.detach().clone() for _ in gw_real))                
                    
                #pbar.close()
                _loss = util.match_loss(gw_syn, gw_real, self.device)
                grad_loss += _loss.item()
                            
                # gradient descent
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                if ol == num_ol - 1:
                    break
                    
                num_il = 5
                synx_in, syny_in = synx.detach(), syny.detach()
                for il in range(num_il):
                    optimizer_model.zero_grad()
                    output_syn_in = _model(synx_in)
                    loss_syn_in = F.mse_loss(output_syn_in, syny_in)
                    loss_syn_in.backward()
                    optimizer_model.step()
                    
            if (i+1) % 10 == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i},. val loss: {val_loss}, test loss: {test_loss}\n")
            else:
                print(f"epoch: {i}, grad loss: {grad_loss/num_ol}")


# python -m DC.distill_grnn -de 0 -e 1000 -sp results/dc_grnn0.0001.txt -lrf 0.01 -lrs 0.0001 -r 1e-3
# python -m DC.distill_grnn -de 1 -e 1000 -sp results/dc_grnn0.01.txt -lrf 0.01 -lrs 0.01 -r 1e-3
# python -m DC.distill_grnn -de 2 -e 1000 -sp results/dc_grnn0.001.txt -lrf 0.01 -lrs 0.001 -r 1e-3
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate')
    parser.add_argument('-r', '--reduction_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')    
    parser.add_argument('-ru', '--rnn_units', type=int, default=2**6, help='rnn hidden unit')
    parser.add_argument('-ed', '--embed_dim', default=10, type=int)
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")

    algo = GradMatch(dataloader, args, device)
    algo.train()