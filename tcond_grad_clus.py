import torch.nn as nn
from tqdm import tqdm
from model import gwnet
import torch
import util
import sys
import copy
import random
from core_set import get_cluster
import numpy as np


class TCondGradClus:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate *  data['train_loader'].xs.shape[0])
        scaler = data['scaler']
        
        # Define condensed data
        xs = data['train_loader'].xs
        ys = data['train_loader'].ys[...,0]
        
        num_total = xs.shape[0]
        mapping = get_cluster(ys, self.num_elems, device)

        _shape = [self.num_elems] + list(xs.shape[1:])
        self.synx = np.zeros(_shape)
        _shape = [self.num_elems] + list(ys.shape[1:])
        self.syny = np.zeros(_shape)

        _cnt = [0 for _ in range(self.num_elems)]
        for i in range(num_total):
            self.synx[mapping[i]] += xs[i]
            self.syny[mapping[i]] += ys[i]
            _cnt[mapping[i]] += 1

        for i in range(self.num_elems):
            if _cnt[i] > 0:
                self.synx[i] /= _cnt[i]
                self.syny[i] /= _cnt[i]

        self.syny = scaler.transform(self.syny)
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)   
        self.idx2cluster = mapping

        self.syn_weight = torch.tensor(_cnt, device=device, dtype=torch.float) / num_total        
        print(f'feat x shape: {self.synx.shape}')
        print(f'feat y shape: {self.syny.shape}')

        # Get dataloader
        self.cluster2idx = [[] for _ in range(self.num_elems)]
        for i in range(num_total):
            self.cluster2idx[self.idx2cluster[i]].append(i)
        
        self.dl_list = []
        for i in range(self.num_elems):
            _loader = util.DataLoader(xs[self.cluster2idx[i]], ys[self.cluster2idx[i]], args.batch_size)
            self.dl_list.append(_loader)
    

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
        _model = nn.DataParallel(_model, device_ids=[0,1,2,3])
        optimizer = torch.optim.Adam(_model.module.parameters(), lr=0.001, weight_decay=0.0001)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            _model.module.train()
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            loss_syn = torch.sum(torch.abs(output_syn - syny), dim=(1, 2))
            loss_syn = torch.sum(loss_syn * self.syn_weight.unsqueeze(1))
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.module.eval()
            if (i+1)%20 == 0:
                with torch.no_grad():
                    val_loss = _model.module.test_model(data['val_loader'], scaler)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.module.state_dict())

        _model.module.load_state_dict(min_params)
        _model.module.eval()
        with torch.no_grad():
            test_loss = _model.module.test_model(data['test_loader'], scaler)

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
        optimizer = torch.optim.Adam([synx, syny], lr=args.learning_rate)
        for i in tqdm(range(args.epochs)):
            _model = gwnet(self.device, num_nodes, 0, in_dim, args.seq_length, 
                               residual_channels=args.nhid, dilation_channels=args.nhid, 
                               skip_channels=8*args.nhid, end_channels=16*args.nhid)
            model_params = list(_model.parameters())
            _model.initialize()
            _model.to(self.device)
            _model = nn.DataParallel(_model, device_ids=[0,1,2,3])
            _model.module.train()

            # compute synthetic gradient
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            loss_syn = torch.mean(torch.abs(output_syn - syny), dim=(1, 2))

            optimizer.zero_grad()
            train_loss = 0
            for j in range(self.num_elems):
                gw_syn = torch.autograd.grad(loss_syn[j], model_params, create_graph=True)
                
                # compute real gradient                    
                num_real = 0
                gw_real = []
                for jj, (x, y) in enumerate(self.dl_list[j].get_iterator()):                
                    # Compute real gradient                            
                    realx = torch.tensor(x, device=self.device, dtype=torch.float)
                    realx = realx.transpose(1, 3)
                    realy = torch.tensor(y, device=self.device, dtype=torch.float)
                    realy = realy[:,:,:]
                    output_real = _model(realx).squeeze()
                    output_real = scaler.inverse_transform(output_real)
                    loss_real, curr_num_real = util.masked_ae(output_real, realy, 0.)

                    if (jj == self.dl_list[j].num_batch - 1):
                        gw_real_curr = torch.autograd.grad(loss_real, model_params)                                    
                    else:
                        gw_real_curr = torch.autograd.grad(loss_real, model_params, retain_graph=True)
                    gw_real_curr = list((_.detach().clone() for _ in gw_real_curr))
                    num_real += curr_num_real
                    
                    for k, gw in enumerate(gw_real_curr):
                        if len(gw_real) < len(gw_real_curr):
                            gw_real.append(gw)
                        else:
                            gw_real[k] = gw_real[k] + gw
                #pbar.update(x.shape[0])

                for k in range(len(gw_real)):
                    gw_real[k] /= num_real
                
            #pbar.close()
                _loss = self.syn_weight[j] * util.match_loss(gw_syn, gw_real, self.device)
                if j < self.num_elems - 1:
                    _loss.backward(retain_graph=True)
                else:
                    _loss.backward()
                train_loss += _loss.item()
            # gradient descent

            optimizer.step()
            if (i+1) % 10 == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, train_loss: {train_loss}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, train_loss: {train_loss}, val loss: {val_loss}, test loss: {test_loss}\n")
            else:
                print(f"epoch: {i}, train loss: {train_loss}")