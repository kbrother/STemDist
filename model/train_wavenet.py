import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.gwave import gwnet
import torch.optim as optim
import math


# python -m model.train_wavenet -de 0 -d ../data/METR-LA-Tensor -lr 1e-2 -e 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA-Tensor', help='data path')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-dr', '--dropout',type=float,default=0.3,help='dropout rate')
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
    model = gwnet(device, num_nodes, args.dropout, in_dim, args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=8*args.nhid, end_channels=16*args.nhid)
    #model = gwnet(device, num_nodes, args.dropout, None, True, True, None, in_dim, args.seq_length, 32, 32, 256, 512, 2)
    model.to(device)
    #model.reset_parameters()
    
    _optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for i in range(args.epochs):

        model.train()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.tensor(x, device=device, dtype=torch.float)
            trainx = trainx.transpose(1, 3)
            trainy = torch.tensor(y, device=device, dtype=torch.float)
            trainy = trainy[:,:,:,0]
            output = model(trainx).squeeze()
            output = scaler.inverse_transform(output)
            curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
            curr_loss /= num_val_entry

            _optimizer.zero_grad()
            curr_loss.backward()
            _optimizer.step()

        model.eval()
        with torch.no_grad():               
            val_mae = model.test_model(dataloader['val_loader'], scaler)
            test_mae = model.test_model(dataloader['test_loader'], scaler)            
            print(f'epoch: {i}, valid mae: {val_mae}, test mae: {test_mae}')