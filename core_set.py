import torch.nn as nn
import torch
import copy
import sys
import util
import random
from model import gwnet
from tqdm import tqdm
import numpy as np


class Coreset:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate *  data['train_loader'].xs.shape[0])


    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx
        syny = self.syny

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']
        _model = gwnet(self.device, num_nodes, args.dropout, in_dim, args.seq_length, 
                           residual_channels=args.nhid, dilation_channels=args.nhid, 
                           skip_channels=8*args.nhid, end_channels=16*args.nhid)
        _model.to(self.device)
        _model = nn.DataParallel(_model, device_ids=[0, 1,2,3])
        optimizer = torch.optim.Adam(_model.module.parameters(), lr=0.001, weight_decay=0.0001)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            _model.module.train()
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            output_syn = scaler.inverse_transform(output_syn)
            loss_syn = util.mae(output_syn, syny)
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


class RandomSample(Coreset):
    def __init__(self, data, args, device):
        super().__init__(data, args, device)
        
        num_total = data['train_loader'].xs.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = self.data['train_loader'].xs[sampled_idx]
        self.syny = self.data['train_loader'].ys[sampled_idx, :, :, 0]
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
    

    def train(self):
        min_i, val_loss, test_loss = self.test_syn()
        print(f"min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        with open(self.args.save_path, 'a') as f:
            f.write(f"min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        


def get_cluster(vecs, num_cents, device):
    num_entry = vecs.shape[0]
    vecs = vecs.reshape(num_entry, -1)    
    _mean = np.mean(vecs, axis=0)
    _std = np.std(vecs, axis=0)
    vecs = (vecs - _mean) / _std

    torch_vecs = torch.tensor(vecs, device=device)
    centroids = 2*(np.random.rand(num_cents, vecs.shape[1]) - 0.5)    
    for i in tqdm(range(90)):
        curr_dist = torch.zeros((num_cents, num_entry), device=device)
        torch_centroids = torch.tensor(centroids, device=device)
        for j in range(num_cents):
            curr_dist[j, :] = torch.sum(torch.square(torch_centroids[j] - torch_vecs), dim=1)

        curr_dist = curr_dist.cpu().numpy()
        curr_dist = curr_dist.transpose()
        mapping = np.argmin(curr_dist, axis=1)
        #if (i+1)%1 == 0:
        #    print(np.sum(np.min(curr_dist,axis=1)))

        _cnt = [0 for _ in range(num_cents)]
        centroids = np.zeros((num_cents, vecs.shape[1]))
        for j in range(num_entry):
            centroids[mapping[j]] += vecs[j]
            _cnt[mapping[j]] += 1
        
        for i in range(num_cents):
            centroids[i] /= _cnt[i]

    return mapping
    

class Kmeans(Coreset):
    def __init__(self, data, args, device):
        super().__init__(data, args, device)
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

        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)

    
    def train(self):
        min_i, val_loss, test_loss = self.test_syn()
        print(f"min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        with open(self.args.save_path, 'a') as f:
            f.write(f"min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        
 
