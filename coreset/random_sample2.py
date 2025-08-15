import argparse
import util
import random
import torch


# python -m coreset.random_sample2 -d ../data/GBA -srr 0.1 -nrr 0.05 -sp results/random_gba.pt
# python -m coreset.random_sample2 -d ../data/GLA -srr 0.1 -nrr 0.05 -sp results/random_gla.pt
# python -m coreset.random_sample2 -d ../data/ERA5 -srr 0.1 -nrr 0.05 -sp results/random_era5.pt
# python -m coreset.random_sample2 -d ../data/CA -srr 0.1 -nrr 0.05 -sp results/random_ca.pt
# python -m coreset.random_sample2 -d ../data/AURORA -srr 0.1 -nrr 0.05 -sp results/random_aurora.pt
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-nrr', '--node_reduction_rate',type=float,default=1e-3,help='learning rate')
    args.add_argument('-srr', '--series_reduction_rate',type=float,default=1e-3,help='learning rate')
    args.add_argument('-sp', '--save_path', type=str, default='results/', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args = args.parse_args()    

    random.seed(args.seed)
    
    data = util.load_dataset(args.data, args.batch_size)
    num_node_total = data['train_loader'].xs.shape[2]
    num_series_total = data['train_loader'].xs.shape[0]
    num_series = round(args.series_reduction_rate * num_series_total)
    num_nodes = round(args.node_reduction_rate * num_node_total)
    
    sampled_idx1 = random.sample(list(range(num_series_total)), num_series)
    sampled_idx2 = random.sample(list(range(num_node_total)), num_nodes)
    synx = data['train_loader'].xs[sampled_idx1][:,:,sampled_idx2,:]
    syny = data['train_loader'].ys[sampled_idx1][:,:,sampled_idx2]
    synx = torch.tensor(synx, dtype=torch.float)
    syny = torch.tensor(syny, dtype=torch.float)
    syny = data['scaler'].transform(syny)

    torch.save({'x': synx, 'y':syny}, args.save_path)
    print(f'x: {synx.shape}, y:{syny.shape}')

