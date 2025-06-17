import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import math
import random


class DataLoader(object):
    def __init__(self, xs, ys, batch_size):
        """
        :param xs:
        :param ys:
        :param batch_size:
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(xs)
        self.num_batch = math.ceil(self.size/self.batch_size)        
        self.xs = xs  # num series x 12 x num nodes x 2
        self.xs_orig = xs
        self.ys = ys


    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

    def get_next(self):
        start_ind = self.batch_size * self.current_ind
        end_ind = self.batch_size * (self.current_ind + 1)
        if end_ind <= self.size:
            x_i = self.xs[start_ind: end_ind, ...]
            y_i = self.ys[start_ind: end_ind, ...]        
            self.current_ind += 1        
        else:
            x_i = self.xs[start_ind: self.size,...]
            y_i = self.ys[start_ind: self.size, ...]        
            self.current_ind = 0
        
        return (x_i, y_i)


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset_old(dataset_dir, batch_size):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'][..., 0]

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
            
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler
    return data


def load_dataset(dataset_dir, batch_size):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'][..., 0]
        
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = None
    return data

    
def masked_mae(preds, labels, null_val):
    mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return torch.mean(loss)


def mae(preds, labels):
    loss = torch.abs(preds - labels)
    return torch.mean(loss)


def mse(preds, labels):
    loss = torch.square(preds - labels)
    return torch.mean(loss)


def masked_ae(preds, labels, null_val):
    mask = (labels != null_val)
    mask = mask.float()    
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return torch.sum(loss), torch.sum(mask)

def masked_se(preds, labels, null_val):
    mask = (labels != null_val)
    mask = mask.float()    
    loss = torch.square(preds - labels)
    loss = loss * mask
    return torch.sum(loss), torch.sum(mask)


def masked_se2(preds, labels, null_val, y_mean):
    mask = (labels != null_val)
    mask = mask.float()    
    loss = torch.square(preds - labels)
    naive_loss = torch.square(y_mean - labels)
    loss = loss * mask
    naive_loss = naive_loss * mask
    return torch.sum(loss), torch.sum(naive_loss)


def distance_wb(gwr, gws):
    shape = gwr.shape


    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    return dis


def distance_wb_taehyung(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        #return torch.tensor(0, dtype=torch.float, device=gwr.device)

    weight = 1/shape[0]
    dis = 1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
    dis = weight * torch.sum(dis)
    
    return dis
    

def match_loss(gw_syn, gw_real, device):
    dis = torch.tensor(0.0).to(device)
    num_entry = 0
    dis_metric = 'ours'
    if dis_metric == 'taehyung':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += torch.numel(gwr) * distance_wb_taehyung(gwr, gws)
            
    elif dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    return dis