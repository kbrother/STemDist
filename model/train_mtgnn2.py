import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.mtgnn_small import gtnet
import torch.optim as optim
import math

# python -m model.train_mtgnn2 -de 1 -d ../data/METR-LA -lr 1e-2 -e 100
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

    '''
    model = gtnet(True, True, 1, num_nodes,
                  device, predefined_A=None,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,
                  conv_channels=32, residual_channels=32,
                  skip_channels=64, end_channels=128,
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)    
    '''

    static_feat = np.load("model/node_embed.npy", allow_pickle=True)
    #print(static_feat)
    static_feat1 = static_feat[()]['v1']
    static_feat2 = static_feat[()]['v2']
    static_feat1 = torch.tensor(static_feat1, device=device, dtype=torch.float)
    static_feat2 = torch.tensor(static_feat2, device=device, dtype=torch.float).transpose(0, 1)
    model = gtnet(True, True, 1, num_nodes,
                  device, predefined_A=None,
                  dropout=0.3, subgraph_size=20, 
                  static_feat1=static_feat1, static_feat2=static_feat2,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)      
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
            val_mse = model.test_model(dataloader['val_loader'], scaler, device)
            test_mse = model.test_model(dataloader['test_loader'], scaler, device)            
            print(f'epoch: {i}, valid mse: {val_mse}, test mse: {test_mse}')