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
import sys


# python -m model.dlinear.train_dlinear -de 0 -d ../data/GLA -lr 1e-2 -e 100 -s 0 -sp results/dlinear-era5.txt
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-sp', '--save_path', type=str, default='results/METR-LA-1e-2', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-bne', '--batch_size_ne', type=int, default=10, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-us', '--use_static_feat', action='store_true', help='true if using node embedding model')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
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
    seq_len = dataloader['train_loader'].xs.shape[1]
    if len(dataloader['train_loader'].ys.shape) == 4:        
        out_dim = dataloader['train_loader'].ys.shape[3]
    else: 
        out_dim = 1

    model = Model(seq_len, seq_len, num_nodes)
    model.to(device)
    #model.reset_parameters()

    _optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    min_loss = sys.float_info.max

    for i in tqdm(range(args.epochs)):
        model.train()        
        dataloader['train_loader'].shuffle()           
        for it, (x, y) in enumerate(dataloader['train_loader'].get_iterator()): 
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
            print(f'epoch: {i},  valid mse: {math.sqrt(val_mse)}, test mse: {math.sqrt(test_mse)}')
            with open(args.save_path, "a") as f:
                f.write(f'epoch: {i},  valid mse: {math.sqrt(val_mse)}, test mse: {math.sqrt(test_mse)}\n')