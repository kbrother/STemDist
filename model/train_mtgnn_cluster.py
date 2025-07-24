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


# python -m model.train_mtgnn_cluster -de 4 -d ../data/GLA -lr 1e-2 -e 100 -s 0 -r 0.1 -ned 128 -b 128 -sp results/gla_cluster.txt
# python -m model.train_mtgnn_cluster -de 0 -d ../data/GBA -lr 1e-2 -e 200 -s 0 -r 0.1 -ned 128 -b 128 -sp results/gba_cluster.txt -s 2
# python -m model.train_mtgnn_cluster -de 3 -d ../data/ERA5_24 -lr 1e-2 -e 100 -s 0 -r 0.1 -ned 32 -b 128 -sp results/era5_cluster.txt
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-sp', '--save_path', type=str, default='results/METR-LA-1e-2', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-bne', '--batch_size_ne', type=int, default=10, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-r', '--ratio', type=float, default=1)
    parser.add_argument('-ned', '--ne_dim',type=int,default=128,help='')
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    start_time = time.time()
    dataloader = util.load_dataset(args.data, args.batch_size, "1", args.ratio, device)
    print(f"load finish, time: {time.time() - start_time}")
    
    scaler = dataloader['scaler']
    seq_len = dataloader['train_loader'].xs.shape[1]
    num_nodes = dataloader['train_loader'].xs_orig.shape[2]
    num_clusters = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]
    if len(dataloader['train_loader'].ys.shape) == 4:        
        out_dim = dataloader['train_loader'].ys.shape[3]
    else: 
        out_dim = 1

    model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim)      
    model.to(device)
    #model.reset_parameters()

    _optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    min_loss = sys.float_info.max

    xx_train = np.mean(dataloader['train_loader'].xs, axis=0)
    xx_train = np.transpose(xx_train, (1, 0, 2))   # num_nodes x 12
    xx_train = torch.tensor(xx_train, dtype=torch.float, device=device)    
    xx_train = torch.reshape(xx_train, (num_clusters, -1))

    xx_test = np.mean(dataloader['train_loader'].xs_orig, axis=0)
    xx_test = np.transpose(xx_test, (1, 0, 2))   # num_nodes x 12
    xx_test = torch.tensor(xx_test, dtype=torch.float, device=device)    
    xx_test = torch.reshape(xx_test, (num_nodes, -1))

    _weight = torch.tensor(dataloader['train_loader'].label_cnt, device=device)
    _weight = _weight / torch.sum(_weight)
    _weight = _weight.unsqueeze(0).unsqueeze(0)
    for i in range(args.epochs):        
        model.train()        
        dataloader['train_loader'].shuffle()           
        for it, (x, y) in enumerate(tqdm(dataloader['train_loader'].get_iterator())):        
            model.embed_forward(xx_train)
            trainx = torch.tensor(x, device=device, dtype=torch.float)                
            trainx = trainx.transpose(1, 3)
            trainy = torch.tensor(y, device=device, dtype=torch.float)          
            output = model(trainx).squeeze()
            output = scaler.inverse_transform(output)
            curr_loss = torch.square(output - trainy) * _weight
            curr_loss = torch.mean(curr_loss)
                        
            _optimizer.zero_grad()
            curr_loss.backward()
            _optimizer.step()

        model.eval()        
        with torch.no_grad():     
            model.embed_forward(xx_test)            
            val_mse = model.test_model(dataloader['val_loader'], scaler, device)
            test_mse = model.test_model(dataloader['test_loader'], scaler, device)    
            print(f'epoch: {i},  valid mse: {math.sqrt(val_mse)}, test mse: {math.sqrt(test_mse)}')
            with open(args.save_path, "a") as f:
                f.write(f'epoch: {i},  valid mse: {math.sqrt(val_mse)}, test mse: {math.sqrt(test_mse)}\n')