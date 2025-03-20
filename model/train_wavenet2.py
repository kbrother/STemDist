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
from scipy.io import loadmat
import sys
import numpy as np
from model.node_embed import NodeEmbedding_rnn


# python -m model.train_wavenet2 -de 3 -d ../data/METR-LA -lr 1e-2 -e 100
# python -m model.train_wavenet -de 0 -d ../data/PEMS-BAY -lr 1e-2 -e 100
if __name__ == "__main__":
    torch.set_num_threads(4)
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
    embedding1 = NodeEmbedding_rnn(12, 256, 10).to(device)
    embedding2 = NodeEmbedding_rnn(12, 256, 10).to(device)
    model = gwnet(device, num_nodes, args.dropout, in_dim, args.seq_length, residual_channels=args.nhid,  
                  dilation_channels=args.nhid, skip_channels=8*args.nhid, end_channels=16*args.nhid,
                  node_vec1=embedding1, node_vec2=embedding2)
    model.to(device)
    #model.reset_parameters()

    params = list(model.parameters()) + list(embedding1.parameters()) + list(embedding2.parameters())
    _optimizer = optim.Adam(params, lr=args.learning_rate)
    min_val_mse = sys.float_info.max
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
            
            model.set_node_embed(train_embed1, train_embed2)   
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
            val_mae = model.test_model2(embedding1, embedding2, dataloader['val_loader'], scaler)
            test_mae = model.test_model2(embedding1, embedding2, dataloader['test_loader'], scaler)            
            print(f'epoch: {i}, valid mae: {val_mae}, test mae: {test_mae}')