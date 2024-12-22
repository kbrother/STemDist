import torch.nn as nn
from tqdm import tqdm
from model import gwnet, ConvNet
import torch
import util
import sys
import copy
import random
import torch.nn.functional as F


class GwaveGradFreq:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate *  data['train_loader'].xs.shape[0])
        scaler = data['scaler']
        
        # Define condensed data
    
        _shape = [self.num_elems] + list(data['train_loader'].xs.shape[1:])
        self.synx = torch.rand(tuple(_shape), device=device, dtype=torch.float)
        # num elems x seq len x num point x feature
        _shape = [self.num_elems] + list(data['train_loader'].ys.shape[1:-1])
        self.syny = torch.rand(_shape, device=device, dtype=torch.float)
        # num elems x seq len x num point
        self.synx = nn.Parameter(self.synx)
        self.syny = nn.Parameter(self.syny)

        print(f'feat x shape: {self.synx.shape}') # [sample 수, timesteps, node 수, feature 개수]
        print(f'feat y shape: {self.syny.shape}') # [sample 수, timesteps, node 수]
 

    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx.detach().clone()
        syny = self.syny.detach().clone()

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']
        _model = gwnet(self.device, num_nodes, args.dropout, in_dim, args.seq_length, 
                           residual_channels=args.nhid, dilation_channels=args.nhid, 
                           skip_channels=8*args.nhid, end_channels=16*args.nhid)
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=1e-4, weight_decay=0.0001)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            _model.train()
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            loss_syn = util.mse(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%20 == 0:
                with torch.no_grad():
                    val_loss = _model.test_model(data['val_loader'], scaler)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            test_loss = _model.test_model(data['test_loader'], scaler)

        return min_i, min_val_loss, test_loss
                
    
    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']

        optimizer = torch.optim.Adam([synx, syny], lr=args.learning_rate)
        for i in tqdm(range(args.epochs)):
            data['train_loader'].shuffle()
            data['train_loader'].current_ind = 0            
            _model = gwnet(self.device, num_nodes, 0, in_dim, args.seq_length, 
                               residual_channels=args.nhid, dilation_channels=args.nhid, 
                               skip_channels=8*args.nhid, end_channels=16*args.nhid)
            model_params = list(_model.parameters())
            _model.initialize()
            _model.to(self.device)
            _model.train()
            optimizer_model = torch.optim.Adam(model_params, lr=0.0001)

            train_loss = 0
            num_ol = 20
            num_real_total = 0
            for ol in range(num_ol):
                # compute synthetic gradient
                output_syn = _model(synx.transpose(1, 3)).squeeze()
                loss_syn = util.mse(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

                # Compute real gradient                            
                x, y = data['train_loader'].get_next()
                real_x = torch.tensor(x, device=self.device, dtype=torch.float)
                realx = real_x.transpose(1, 3)
                realy = torch.tensor(y, device=self.device, dtype=torch.float)
                realy = realy[:,:,:,0]
                output_real = _model(realx).squeeze()
                output_real = scaler.inverse_transform(output_real)
                loss_real, num_real = util.masked_se(output_real, realy, 0.)
                gw_real = torch.autograd.grad(loss_real/num_real, model_params)
                gw_real = list((_.detach().clone() for _ in gw_real))                
                    
                #pbar.close()
                _loss = util.match_loss(gw_syn, gw_real, self.device)
                train_loss += loss_real.item()
                num_real_total += num_real

                ### frequency domain matching
                # compute frequency
                syn_f = torch.fft.rfft(synx, dim=-1)
                syn_f = torch.view_as_real(syn_f).reshape(syn_f.shape[0], syn_f.shape[1], syn_f.shape[2], -1)
                syn_f = syn_f[:,:,:, :x.shape[-1]]
                # print(syn_f.shape)  

                real_f = torch.fft.rfft(real_x, dim=-1)
                real_f = torch.view_as_real(real_f).reshape(real_f.shape[0], real_f.shape[1], real_f.shape[2], -1)
                real_f = real_f[:,:,:, :x.shape[-1]]
                # print(real_f.shape) 

                syn_f_mean = syn_f.mean(dim=0)   # [12, 207, 2]
                real_f_mean = real_f.mean(dim=0) 
                syn_f_flat = syn_f_mean.flatten()  # [12 * 207 * 2]
                real_f_flat = real_f_mean.flatten()

                cosine_sim = F.cosine_similarity(syn_f_flat.unsqueeze(0), real_f_flat.unsqueeze(0))  
                loss_freq = -cosine_sim.mean()  
                
                _loss_final = _loss + loss_freq
                
                # gradient descent
                optimizer.zero_grad()
                _loss_final.backward()
                # _loss.backward()
                optimizer.step()

                if ol == num_ol - 1:
                    break
                    
                num_il = 5
                synx_in, syny_in = synx.detach(), syny.detach()
                for il in range(num_il):
                    optimizer_model.zero_grad()
                    output_syn_in = _model(synx_in.transpose(1,3)).squeeze()
                    loss_syn_in = util.mse(output_syn_in, syny_in)
                    loss_syn_in.backward()
                    optimizer_model.step()
                    
            if (i+1) % 10 == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, train_loss: {train_loss/num_real_total}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, train_loss: {train_loss/num_real_total}, val loss: {val_loss}, test loss: {test_loss}\n")
            else:
                print(f"epoch: {i}, train loss: {train_loss/num_real_total}")
                