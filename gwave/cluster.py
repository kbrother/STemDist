import argparse
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
from model import gwnet
import torch


# python gwave/cluster.py --use_embed -sp mapping/METR-LA2.npy
# python gwave/cluster.py -sp mapping/METR-LA.npy
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_embed", action="store_true")
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--buffer', type=str, default='../data/params/METR-LA/replay_buffer_0.pt', help='data path')
    parser.add_argument('-n', '--num_cents', type=int, default=5)  
    parser.add_argument('-s', '--seed', type=int, default=0, help='')    
    parser.add_argument('-sp', '--save_path', type=str, default='mapping/METR-LA.npy')
    args = parser.parse_args()

    # random seed setting    
    cat_data = np.load(os.path.join(args.data, 'train.npz'))
    num_nodes = cat_data['x'].shape[2]
    in_dim = cat_data['x'].shape[3]
    if args.use_embed:
        _model = gwnet(torch.device("cpu"), num_nodes, 0, in_dim, 12, 
                           residual_channels=32, dilation_channels=32, 
                           skip_channels=8*32, end_channels=16*32)
        traj = torch.load(args.buffer)        

        with torch.no_grad():
            for param, loaded_param in zip(_model.parameters(), traj[-1]):
                param.data.copy_(loaded_param)
        full_data = _model.nodevec1.detach().numpy()
    else:
        cat_data = np.load(os.path.join(args.data, 'train.npz'))
        full_data = np.concatenate((cat_data['x'], cat_data['y']), axis=1)
        print(full_data.shape)
        full_data = np.transpose(full_data, (2, 0, 1, 3))   # num point x num data x seq len x feature 
        
    print("load finish")

    num_points = full_data.shape[0]
    full_data = full_data.reshape(num_points, -1)
    kmeans = KMeans(n_clusters=args.num_cents, random_state=args.seed).fit(full_data)
    labels = kmeans.labels_
    print(labels)
    
    _cnt = [0 for _ in range(args.num_cents)]
    for p in range(num_points):
        _cnt[labels[p]] += 1

    _cnt = [c/num_points for c in _cnt]        
    print(_cnt)

    np.save(args.save_path, labels)
    