import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.mtgnn import gtnet
import torch.optim as optim
import math
import torch.nn as nn


# python -m condTSF.buffer_mtgnn -de 0 -d ../data/METR-LA -lr 1e-3 -e 10 -s 0
if __name__ == "__main__":
    torch.set_num_threads(3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-e', '--epochs',type=int,default=10,help='')
    parser.add_argument('-lr', '--lr', default=0.003, type=float)
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='../data/params/METR-LA-mtgnn/')
    parser.add_argument('-ne', '--num_experts', type=int, default=5)
    parser.add_argument('-m', '--mom', type=float, default=0.9, help='momentum')
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")

    scaler = dataloader['scaler']
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]

    for it in range(args.num_experts):
        model = gtnet(True, True, 2, num_nodes,
                  device, predefined_A=None,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,
                  conv_channels=32, residual_channels=32,
                  skip_channels=64, end_channels=128,
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
        model = model.to(device)        
        curr_traj = [[p.clone().detach().cpu() for p in model.parameters()]]        
        _optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
        for i in range(args.epochs):
            model.train()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(tqdm(dataloader['train_loader'].get_iterator(), desc="Processing")):
                trainx = torch.tensor(x, device=device, dtype=torch.float)
                trainy = torch.tensor(y, device=device, dtype=torch.float)
                trainy = trainy[:,:,:,0]
                output = model(trainx.transpose(1,3)).squeeze()
                output = scaler.inverse_transform(output)
                curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)

                if num_val_entry > 0:
                    curr_loss /= num_val_entry
                    _optimizer.zero_grad()
                    curr_loss.backward()
                    _optimizer.step()
    
            model.eval()
            with torch.no_grad():               
                val_mse = model.test_model(dataloader['val_loader'], scaler, device)
                test_mse = model.test_model(dataloader['test_loader'], scaler, device)            
                print(f'epoch: {i}, valid mse: {val_mse}, test mse: {test_mse}')
                with open(args.save_path + f"replay_buffer_{args.num_experts*args.seed + it}.txt", "a") as f:
                    f.write(f'epoch: {i}, valid mse: {val_mse}, test mse: {test_mse}\n')

            curr_traj.append([p.clone().detach().cpu() for p in model.parameters()])        
        torch.save(curr_traj, args.save_path + f"replay_buffer_{args.num_experts*args.seed + it}.pt")
        