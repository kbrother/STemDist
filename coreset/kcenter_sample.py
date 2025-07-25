import argparse
import util
import random
import torch
import numpy as np


def kcenter_selection(xs, ys, k, device='cpu'):
    xs_flat = xs.reshape(xs.shape[0], -1)
    ys_flat = ys.reshape(ys.shape[0], -1)
    
    features = np.concatenate([xs_flat, ys_flat], axis=1)
    features = torch.tensor(features, dtype=torch.float, device=device)
    
    N = features.shape[0]
    selected_indices = []
    
    first_center = torch.randint(0, N, (1,)).item()
    selected_indices.append(first_center)
    
    for i in range(k - 1):
        unselected_mask = torch.ones(N, dtype=torch.bool, device=device)
        unselected_mask[selected_indices] = False
        unselected_indices = torch.where(unselected_mask)[0]        
        if len(unselected_indices) == 0:
            break
        
        selected_features = features[selected_indices]
        unselected_features = features[unselected_indices]
        distances = torch.cdist(unselected_features, selected_features)
        min_distances, _ = torch.min(distances, dim=1)
        
        next_center_idx = torch.argmax(min_distances).item()
        next_center = unselected_indices[next_center_idx].item()
        selected_indices.append(next_center)
    
    print(f"K-center selection completed. Selected {len(selected_indices)} samples.")
    return selected_indices


# python -m coreset.kcenter_sample -d ../data/GBA -rr 0.01 -sp results/kcenter_gba.pt
# python -m coreset.kcenter_sample -d ../data/GLA -rr 0.01 -sp results/kcenter_gla.pt
# python -m coreset.kcenter_sample -d ../data/ERA5 -rr 0.01 -sp results/kcenter_era5.pt
# python -m coreset.kcenter_sample -d ../data/CA -rr 0.01 -sp results/kcenter_ca.pt
# python -m coreset.kcenter_sample -d ../data/AURORA -rr 0.01 -sp results/kcenter_aurora.pt

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-rr', '--reduction_rate',type=float,default=1e-3,help='reduction rate')
    args.add_argument('-sp', '--save_path', type=str, default='results/', help='save path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    args.add_argument('-de', '--device', type=str, default='cpu', help='device to use')
    args = args.parse_args()    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    data = util.load_dataset(args.data, args.batch_size)
    num_total = data['train_loader'].xs.shape[0]
    num_elems = round(args.reduction_rate * num_total)
    
    xs = data['train_loader'].xs
    ys = data['train_loader'].ys
    sampled_idx = kcenter_selection(xs, ys, num_elems, args.device)
    
    synx = data['train_loader'].xs[sampled_idx]
    syny = data['train_loader'].ys[sampled_idx]
    synx = torch.tensor(synx, dtype=torch.float)
    syny = torch.tensor(syny, dtype=torch.float)
    syny = data['scaler'].transform(syny)

    torch.save({'x': synx, 'y': syny}, args.save_path)
    print(f'x: {synx.shape}, y:{syny.shape}')