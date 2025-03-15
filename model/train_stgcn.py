import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.stgcn import STGCN
from model.node_embed import NodeEmbedding
import torch.optim as optim
import math
import sys
import torch.nn.functional as F


# python -m model.train_stgcn -de 0 -d ../data/METR-LA -lr 1e-2 -e 100
# python -m model.train_mtgnn2 -d ../data/PEMS-BAY -de 0 -lr 1e-2 -e 100
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
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

    embedding1 = NodeEmbedding(12, 256, 10).to(device)
    embedding2 = NodeEmbedding(12, 256, 10).to(device)
    model = STGCN(num_nodes, in_dim, 12, 12) 
    model.to(device)
    #model.reset_parameters()

    params = list(model.parameters()) + list(embedding1.parameters()) + list(embedding2.parameters())
    _optimizer = optim.Adam(params, lr=args.learning_rate)
    min_loss = sys.float_info.max
    for i in range(args.epochs):
        model.train()
        embedding1.train()
        embedding2.train()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(tqdm(dataloader['train_loader'].get_iterator())):
            trainx = torch.tensor(x, device=device, dtype=torch.float)
            train_embed1 = embedding1(trainx[...,0])
            train_embed1 = torch.mean(train_embed1, dim=0)
            train_embed2 = embedding2(trainx[...,0])
            train_embed2 = torch.mean(train_embed2, dim=0)
            
            adj = F.relu(torch.tanh(torch.mm(train_embed1, train_embed2.t())))    
            trainx = trainx.transpose(1, 2)
            trainy = torch.tensor(y, device=device, dtype=torch.float)
            trainy = trainy[:,:,:,0]
            trainy =  trainy.transpose(1,2)
            output = model(adj, trainx)
            output = scaler.inverse_transform(output)
            curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
            curr_loss /= num_val_entry

            _optimizer.zero_grad()
            curr_loss.backward()
            _optimizer.step()

        model.eval()
        embedding1.eval()
        embedding2.eval()
        with torch.no_grad():               
            val_mse = model.test_model(embedding1, embedding2, dataloader['val_loader'], scaler, device)
            test_mse = model.test_model(embedding1, embedding2, dataloader['test_loader'], scaler, device)            
            print(f'epoch: {i}, valid mse: {val_mse}, test mse: {test_mse}')
