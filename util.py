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
from sklearn.metrics import pairwise_distances
from shapely.geometry import box
import geopandas as gpd
from sklearn.decomposition import PCA
import pandas as pd
import pygeoda


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


def robust_kmeans(xs, num_clusters, num_series, seq_len, num_feat, device='cuda'):
    input_xs = torch.tensor(xs, device=device, dtype=torch.float)
    kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', init_method="kmeans++")
    labels = kmeans.fit_predict(input_xs).cpu().numpy()
    centroids = kmeans.centroids.cpu().numpy()  # (K, D)
    counts = np.bincount(labels, minlength=num_clusters)
    for empty_cluster in np.where(counts == 0)[0]:
        largest_cluster = np.argmax(counts)
        idx_in_largest = np.where(labels == largest_cluster)[0]
        largest_points = input_xs[idx_in_largest].cpu().numpy()    # 해당 클러스터 내에서 centroid와 가장 먼 점 찾기
        centroid_largest = centroids[largest_cluster:largest_cluster+1]
        dists = pairwise_distances(largest_points, centroid_largest)
        farthest_idx_local = np.argmax(dists)
        farthest_idx = idx_in_largest[farthest_idx_local]

        labels[farthest_idx] = empty_cluster
        counts[largest_cluster] -= 1
        counts[empty_cluster] += 1

    centroids = []
    for k in range(num_clusters):
        member_idx = np.where(labels == k)[0]
        centroid_k = np.mean(xs[member_idx], axis=0)
        centroids.append(centroid_k)
    centroids = np.stack(centroids, axis=0)
    xs_clustered = np.reshape(centroids, (num_clusters, num_series, seq_len, num_feat))
    xs_clustered = np.transpose(xs_clustered, (1, 2, 0, 3))  # [num_series, seq_len, num_clusters, num_feat]

    return xs_clustered, labels


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

        xs_clustered, labels = robust_kmeans(xs, num_clusters, num_series, seq_len, num_feat, device)
        self.xs = xs_clustered        
        self.ys = np.zeros((num_series, seq_len, num_clusters))
        label_cnt = [0 for _ in range(num_clusters)]
        for i in range(num_nodes):
            self.ys[:, :, labels[i]] = self.ys[:, :, labels[i]] + ys[:, :, i]
            label_cnt[labels[i]] += 1

        self.label_cnt = label_cnt
        for i in range(num_clusters):
            # if label_cnt[i] == 0:
            #     print(f"!:label_cnt[{i}] is 0, num_clusters: {num_clusters}, num_nodes: {num_nodes}")
            #     self.ys[:,:,i] = np.zeros((num_series, seq_len))
            # else:
            self.ys[:,:,i] = self.ys[:,:,i]/label_cnt[i]


class DataLoaderRedcap(DataLoader):
    def __init__(self, xs, ys, batch_size, num_clusters, nrows, ncols):
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(xs)
        self.num_batch = math.ceil(self.size/self.batch_size)     

        self.xs_orig = xs  # num sereis x 12 x num nodes x num_feat
        num_nodes, num_series = xs.shape[2], xs.shape[0]
        seq_len, num_feat = xs.shape[1], xs.shape[3]

        labels = self.redcap(xs, num_clusters, nrows, ncols)
        label_cnt = [0 for _ in range(num_clusters)]
        self.xs = np.zeros((num_series, seq_len, num_clusters, num_feat))        
        self.ys = np.zeros((num_series, seq_len, num_clusters))
        for i in range(num_nodes):
            self.ys[:, :, labels[i]] = self.ys[:, :, labels[i]] + ys[:, :, i]
            self.xs[:, :,  labels[i], :] = self.xs[:,:,labels[i],:] + xs[:,:,i,:]
            label_cnt[labels[i]] += 1

        self.label_cnt = label_cnt
        for i in range(num_clusters):
            assert label_cnt[i] > 0
            self.xs[:,:,i,:] = self.xs[:,:,i,:]/label_cnt[i]
            self.ys[:,:,i] = self.ys[:,:,i]/label_cnt[i]
            
        
    def redcap(self, xs, num_clusters, nrows, ncols):
        num_nodes, num_series = xs.shape[2], xs.shape[0]
        seq_len, num_feat = xs.shape[1], xs.shape[3]
        
        geoms = []
        ids = []
        cell_size = 1.0
        for r in range(nrows):
            for c in range(ncols):
                xmin, ymin = c * cell_size, r * cell_size
                xmax, ymax = xmin + cell_size, ymin + cell_size
                geoms.append(box(xmin, ymin, xmax, ymax))
                ids.append(r * ncols + c)

        gdf = gpd.GeoDataFrame({"cell_id": ids}, geometry=geoms, crs="EPSG:3857")
        _data = xs[:, 0, :, :] # num series x num nodes x num feat
        _data = np.transpose(_data, (1, 0, 2))
        _data = np.reshape(_data, (num_nodes, -1))        
        Xp = PCA(n_components=50).fit_transform(_data)
        dfX = pd.DataFrame(Xp, columns=[f"f{i}" for i in range(50)])
        gda = pygeoda.open(gdf) 
        w = pygeoda.rook_weights(gda)
        method = "fullorder-wardlinkage"

        res = pygeoda.redcap(num_clusters, w, dfX, method)
        return np.asarray(res["Clusters"])-1
        
    
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


def load_dataset(dataset_dir, batch_size, loader_type=None, _ratio=1.0, device=None, nrows=None, ncols=None):
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
        if loader_type == "kmeans":
            data['train_loader'] = DataLoaderCluster(data['x_train'], data['y_train'], batch_size, num_clusters, device)          
        elif loader_type == "redcap":
            data['train_loader'] = DataLoaderRedcap(data['x_train'], data['y_train'], batch_size, num_clusters, nrows, ncols)       
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


def masked_ae2(preds, labels, null_val, y_mean):
    mask = (labels != null_val)
    mask = mask.float()    
    loss = torch.abs(preds - labels)
    naive_loss = torch.abs(y_mean - labels)
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