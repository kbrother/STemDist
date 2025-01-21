import argparse
import util
from model.glinear import GLinear
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import random
import torch
import numpy as np
import sys


# python -m model.train_glinear -de 0 -lr 0.01 -e 30
if __name__ == "__main__":
    torch.set_num_threads(4)
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('-de', '--device', type=int, default=0, help='')
    args.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-r', '--rnn_units', type=int, default=2**6, help='rnn hidden unit')
    args.add_argument('-ed', '--embed_dim', default=10, type=int)
    args.add_argument('-lr', '--lr', default=0.003, type=float)
    args.add_argument('-e', '--epochs', default=100, type=int)
    args.add_argument('-s', '--seed', type=int, default=0, help='')
    args.add_argument('-sp', '--save_path', type=str, default='../data/params/METR-LA-glinear/')
    args = args.parse_args()    

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
    model = GLinear(args, in_dim)
    #init loss function, optimizer    

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    model = model.to(device)
    '''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    '''
    
    min_val_mse = sys.float_info.max
    for e in range(args.epochs):
        model.train()
        train_loss, num_entry = 0, 0
        dataloader['train_loader'].shuffle()
        
        for iter, (x, y) in enumerate(tqdm(dataloader['train_loader'].get_iterator())):
            trainx = torch.tensor(x, device=device, dtype=torch.float)
            trainy = torch.tensor(y, device=device, dtype=torch.float)
            trainy = trainy[:,:,:,0]
            output = model(trainx)
            output = scaler.inverse_transform(output)
            curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)            
            curr_loss /= num_val_entry

            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():               
            val_mse = model.test_model(dataloader['val_loader'], scaler, device)
            test_mse = model.test_model(dataloader['test_loader'], scaler, device)            
            print(f'epoch: {e}, valid mse: {val_mse}, test mse: {test_mse}')

            if min_val_mse > val_mse:
                min_val_mse = val_mse
                #final_embed = model.node_embeddings.data.clone().detach().cpu()