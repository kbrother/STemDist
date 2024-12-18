import argparse
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.cluster import KMeans

# python cluster.py 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-n', '--num_cents', type=int, default=10)  
    parser.add_argument('-s', '--seed', type=int, default=0, help='')    
    args = parser.parse_args()

    # random seed setting    
    cat_data = np.load(os.path.join(args.data, 'train.npz'))
    full_data = np.concatenate((cat_data['x'], cat_data['y']), axis=1)
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

    
    