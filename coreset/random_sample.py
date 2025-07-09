import argparse
import util
import random
import torch


# python -m coreset.random_sample -d ../data/GBA -rr 0.01 -sp results/random_gba.pt
# python -m coreset.random_sample -d ../data/GLA -rr 0.01 -sp results/random_gla.pt
# python -m coreset.random_sample -d ../data/ERA5 -rr 0.01 -sp results/random_era5.pt
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-rr', '--reduction_rate',type=float,default=1e-3,help='learning rate')
    args.add_argument('-sp', '--save_path', type=str, default='results/', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args = args.parse_args()    

    random.seed(args.seed)
    
    data = util.load_dataset(args.data, args.batch_size)
    num_total = data['train_loader'].xs.shape[0]
    num_elems = round(args.reduction_rate * num_total)
    sampled_idx = random.sample(list(range(num_total)), num_elems)
    synx = data['train_loader'].xs[sampled_idx]
    syny = data['train_loader'].ys[sampled_idx]
    synx = torch.tensor(synx, dtype=torch.float)
    syny = torch.tensor(syny, dtype=torch.float)
    syny = data['scaler'].transform(syny)

    torch.save({'x': synx, 'y':syny}, args.save_path)
    print(f'x: {synx.shape}, y:{syny.shape}')

