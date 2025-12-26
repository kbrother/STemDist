import torch
import numpy as np
import argparse
import util
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch.optim as optim
from model.mtgnn import gtnet
import torch.optim as optim
import math
import sys

# python -m MTT.buffer_mtgnn -de 7 -lr 1e-3 -e 40 -s 0 -ne 10 -n 0 -d ../data/GBA -sp ../data/params/GBA-MTGNN-2/
# python -m MTT.buffer -de 3 -lr 1e-3 -e 40 -s 0 -ne 10 -n 0 -d ../data/GLA -sp ../data/params/GLA-DLinear/
# python -m MTT.buffer -de 3 -lr 1e-3 -e 40 -s 0 -ne 10 -n 0 -d ../data/CA -sp ../data/params/CA-DLinear/
# python -m MTT.buffer -de 3 -lr 1e-3 -e 40 -s 0 -ne 10 -n 0 -d ../data/ERA5 -sp ../data/params/ERA5-DLinear/
# python -m MTT.buffer -de 2 -lr 1e-3 -e 40 -s 0 -ne 10 -n 0 -d ../data/AURORA -sp ../data/params/AURORA-DLinear/


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-sp', '--save_path', type=str, default='results/METR-LA-1e-2', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-bne', '--batch_size_ne', type=int, default=10, help='batch size')
    parser.add_argument('-sl', '--seq_len', type=int, default=12, help='sequence length')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-us', '--use_static_feat', default=True, action='store_true', help='true if using node embedding model')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-ned', '--ne_dim',type=int,default=32,help='')
    parser.add_argument('-ne', '--num_experts', type=int, default=5)
    parser.add_argument('-n', '--number', type=int, default=20)
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
    if len(dataloader['train_loader'].ys.shape) == 4:        
        out_dim = dataloader['train_loader'].ys.shape[3]
    else: 
        out_dim = 1
    seq_len = dataloader['train_loader'].xs.shape[1]

    for it in range(args.num_experts):
        model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim) 
        model.to(device)
        #model.reset_parameters()

        curr_traj = [[p.detach().cpu() for p in model.parameters()]]        
        _optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        nm_input_real = np.mean(dataloader['train_loader'].xs, axis=0)  # seq_length x num nodes x in_dim
        nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
        nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
        nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim

        for i in tqdm(range(args.epochs)):
            model.train()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.tensor(x, device=device, dtype=torch.float)   # batch x num seq x num node x 2        
                trainy = torch.tensor(y, device=device, dtype=torch.float)          
                model.embed_forward(nm_input_real)
                output = model(trainx.transpose(1, 3)).squeeze()
                output = scaler.inverse_transform(output)
                curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
                curr_loss /= num_val_entry

                _optimizer.zero_grad()
                curr_loss.backward()
                _optimizer.step()
    
            model.eval()
            with torch.no_grad():
                model.embed_forward(nm_input_real)
                val_mse = math.sqrt(model.test_model(dataloader['val_loader'], scaler, device))
                test_mse = math.sqrt(model.test_model(dataloader['test_loader'], scaler, device))            
                print(f'epoch: {i}, valid mse: {val_mse}, test mse: {test_mse}')

            curr_traj.append([p.detach().cpu() for p in model.parameters()])        
        torch.save(curr_traj, args.save_path + f"replay_buffer_{args.number + it}.pt")
        