import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.dlinear.dlinear import Model
import torch.optim as optim
import math
import torch.nn as nn


# python -m CondTSF.buffer_dlinear -d ../data/AURORA_final -de 0 -lr 1e-2 -s 0 -e 15
# python -m CondTSF.buffer_dlinear -d ../data/AURORA_final -de 1 -lr 1e-2 -s 1 -e 15
# python -m CondTSF.buffer_dlinear -d ../data/AURORA_final -de 2 -lr 1e-2 -s 2 -e 15
# python -m CondTSF.buffer_dlinear -d ../data/AURORA_final -de 3 -lr 1e-2 -s 3 -e 15
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/AURORA_final', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-e', '--epochs',type=int,default=10,help='')
    parser.add_argument('-lr', '--lr', default=0.001, type=float)
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='../data/params/AURORA-Dlinear/')
    parser.add_argument('-ne', '--num_experts', type=int, default=5)
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
    seq_len = dataloader['train_loader'].xs.shape[1]

    for it in range(args.num_experts):
        model = Model(seq_len, seq_len, num_nodes)
        model = model.to(device)        
        curr_traj = [[p.clone().detach().cpu() for p in model.parameters()]]        
        _optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for i in range(args.epochs):
            model.train()
            dataloader['train_loader'].shuffle()
            for _, (x, y) in enumerate(dataloader['train_loader'].get_iterator()): 
                trainx = torch.tensor(x, device=device, dtype=torch.float)   # batch x num seq x num node x 2        
                trainx = trainx[..., 0]
                trainy = torch.tensor(y, device=device, dtype=torch.float)          
                output = model(trainx)
                output = scaler.inverse_transform(output)
                curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
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
        