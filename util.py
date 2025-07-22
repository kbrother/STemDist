import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import math
import random
from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans


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
        self.ys_orig = ys
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
            self.shuffle()
            
        return (x_i, y_i)


class DataLoaderCluster2(DataLoader):
    def __init__(self, xs, ys, batch_size, num_clusters, device):
        super().__init__(xs, ys, batch_size)
        kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', init_method="kmeans++")
        xs = np.transpose(xs, (2, 0, 1, 3))  # num nodes x num sereis x 12 x 2
        num_nodes, num_series = xs.shape[0], xs.shape[1]
        seq_len, num_feat = xs.shape[2], xs.shape[3]
        xs = np.reshape(xs, (num_nodes, -1))   # num nodes x ... 
        input_xs = torch.tensor(xs, device=device, dtype=torch.float)
        labels = kmeans.fit_predict(input_xs).cpu().numpy()        
        xs = kmeans.centroids.cpu().numpy()  # num nodes x ...
        xs = np.reshape(xs, (num_clusters, num_series, seq_len, num_feat))
        xs = np.transpose(xs, (1,2,0,3))        
        self.xs_reduced = xs   
        
        self.ys_reduced = np.zeros((num_series, seq_len, num_clusters))
        self.label_cnt = [0 for _ in range(num_clusters)]
        for i in range(num_nodes):
            self.ys_reduced[:, :, labels[i]] = self.ys_reduced[:, :, labels[i]] + ys[:, :, i]
            self.label_cnt[labels[i]] += 1
        for i in range(num_clusters):
            self.ys_reduced[:,:,i] = self.ys_reduced[:,:,i]/self.label_cnt[i]
        
        self.node2cluster = [-1 for _ in range(num_nodes)]
        for n, l in enumerate(labels):            
            self.node2cluster[n] = l
        self.node2cluster = np.array(self.node2cluster)


class DataLoaderCluster(DataLoader):
    def __init__(self, xs, ys, batch_size, num_clusters, device):
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(xs)
        self.num_batch = math.ceil(self.size/self.batch_size)        

        # num series x 12 x num nodes x 2
        self.xs_orig = xs
        xs = np.transpose(xs, (2, 0, 1, 3))  # num nodes x num sereis x 12 x 2
        num_nodes, num_series = xs.shape[0], xs.shape[1]
        seq_len, num_feat = xs.shape[2], xs.shape[3]
        xs = np.reshape(xs, (num_nodes, -1))   # num nodes x ... 

        kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', init_method="kmeans++")
        input_xs = torch.tensor(xs, device=device, dtype=torch.float)
        labels = kmeans.fit_predict(input_xs).cpu().numpy()        
        xs = kmeans.centroids.cpu().numpy()  # num nodes x ...
        xs = np.reshape(xs, (num_clusters, num_series, seq_len, num_feat))
        xs = np.transpose(xs, (1,2,0,3))
        
        self.xs = xs        
        self.ys = np.zeros((num_series, seq_len, num_clusters))
        label_cnt = [0 for _ in range(num_clusters)]
        for i in range(num_nodes):
            self.ys[:, :, labels[i]] = self.ys[:, :, labels[i]] + ys[:, :, i]
            label_cnt[labels[i]] += 1

        self.label_cnt = label_cnt
        for i in range(num_clusters):
            self.ys[:,:,i] = self.ys[:,:,i]/label_cnt[i]

            
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


def load_dataset(dataset_dir, batch_size, loader_type=None, _ratio=1.0, device=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'][..., 0]

    num_feat = data['x_train'].shape[-1]
    scaler_list = []
    for i in range(num_feat):
        scaler_list.append(StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std()))    
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., i] = scaler_list[i].transform(data['x_' + category][..., i])

    if loader_type is not None:
        num_clusters = round(_ratio * data['x_train'].shape[2])

        if loader_type == "1":
            data['train_loader'] = DataLoaderCluster(data['x_train'], data['y_train'], batch_size, num_clusters, device)
        else:
            data['train_loader'] = DataLoaderCluster2(data['x_train'], data['y_train'], batch_size, num_clusters, device)
    else:
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)

    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler_list[0]
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