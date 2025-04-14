import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.mtgnn import gtnet
from model.node_embed import NodeEmbedding_rnn, NodeEmbedding_birnn, NodeEmbedding_attn
import torch.optim as optim
import math
import sys


# python -m model.train_mtgnn2 -de 7 -d ../data/METR-LA -lr 1e-3 -e 100
# python -m model.train_mtgnn2 -d ../data/PEMS-BAY -de 0 -lr 1e-2 -e 100
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-bne', '--batch_size_ne', type=int, default=10, help='batch size')
    parser.add_argument('-sl', '--seq_len', type=int, default=12*24*7, help='sequence length')
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

    max_seq = max(args.seq_len, dataloader['test_loader'].xs.shape[0])
    #embedding1 = NodeEmbedding_rnn(12, 512, 10).to(device)
    #embedding2 = NodeEmbedding_rnn(12, 512, 10).to(device)
    embedding1 = NodeEmbedding_birnn(12, 512, 10).to(device)
    embedding2 = NodeEmbedding_birnn(12, 512, 10).to(device)
    #embedding1 = NodeEmbedding_attn(12, 512, 10).to(device)
    #embedding2 = NodeEmbedding_attn(12, 512, 10).to(device)
    
    model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)      
    model.to(device)
    #model.reset_parameters()

    params = list(model.parameters()) + list(embedding1.parameters()) + list(embedding2.parameters())
    _optimizer = optim.Adam(params, lr=args.learning_rate)
    min_loss = sys.float_info.max

    xx = np.mean(dataloader['train_loader'].xs_orig, axis=0)   # 12 x num nodes x 2
    xx = np.transpose(xx, (1, 0, 2))[:,:,0]   # num_nodes x 12
    xx = torch.tensor(xx, dtype=torch.float, device=device)    
    for i in range(args.epochs):
        model.train()
        embedding1.train()
        embedding2.train()
        dataloader['train_loader'].shuffle()           
        for it, (x, y) in enumerate(tqdm(dataloader['train_loader'].get_iterator())): 
            train_embed1 = embedding1(xx)
            train_embed2 = embedding2(xx)
            
            model.set_node_embed(train_embed1, train_embed2) 
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
        embedding1.eval()
        embedding2.eval()
        with torch.no_grad():     
            test_embed1 = embedding1(xx)
            test_embed2 = embedding2(xx)
            model.set_node_embed(test_embed1, test_embed2) 
            
            val_mse = model.test_model(dataloader['val_loader'], scaler, device)
            test_mse = model.test_model(dataloader['test_loader'], scaler, device)    
            print(f'epoch: {i},  valid mse: {val_mse}, test mse: {test_mse}')