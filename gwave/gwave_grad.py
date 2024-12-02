import torch.nn as nn
from tqdm import tqdm
from model import gwnet
import torch
import util
import sys
import copy
import random


class GwaveGrad:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate *  data['train_loader'].xs.shape[0])
        scaler = data['scaler']
        
        # Define condensed data
    
        _shape = [self.num_elems] + list(data['train_loader'].xs.shape[1:])
        self.synx = torch.rand(tuple(_shape), device=device, dtype=torch.float)
        _shape = [self.num_elems] + list(data['train_loader'].ys.shape[1:-1])
        self.syny = torch.rand(_shape, device=device, dtype=torch.float)
        '''
        num_total = data['train_loader'].xs.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = self.data['train_loader'].xs[sampled_idx]     
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        
        self.syny = self.data['train_loader'].ys[sampled_idx, :, :, 0]
        self.syny = scaler.transform(self.syny)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
        '''
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

        min_i, val_loss, test_loss = self.test_syn()
        print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        with open(args.save_path, 'a') as f:
            f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        
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
                #pbar = tqdm(total=data['train_loader'].xs.shape[0])
                # compute synthetic gradient
                output_syn = _model(synx.transpose(1, 3)).squeeze()
                loss_syn = util.mse(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

                # Compute real gradient                            
                x, y = data['train_loader'].get_next()
                realx = torch.tensor(x, device=self.device, dtype=torch.float)
                realx = realx.transpose(1, 3)
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
                