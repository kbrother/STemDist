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
import copy

# python -m model.train_mtgnn -de 4 -d ../data/GBA_24 -sp results/mtgnn_gba -b 128 -lr 1e-3 
# python -m model.train_mtgnn -de 5 -d ../data/ERA5_24 -sp results/mtgnn_era5 -b 32 -lr 1e-3
# python -m model.train_mtgnn -de 7 -d ../data/GLA_24 -sp results/mtgnn_gla.pt -b 128 -lr 1e-3
# python -m model.train_mtgnn -de 5 -d ../data/AURORA -sp results/mtgnn_aurora.pt -b 32 -lr 1e-3
# python -m model.train_mtgnn -de 3 -d ../data/CA -sp results/mtgnn_ca.pt -b 32 -lr 1e-3
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-sp', '--save_path', type=str, default='results/METR-LA-1e-2', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
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

    model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=args.use_static_feat,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)      
    model.to(device)
    #model.reset_parameters()

    _optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    min_loss = sys.float_info.max

    xx = np.mean(dataloader['train_loader'].xs_orig, axis=0)
    xx = np.transpose(xx, (1, 0, 2))   # num_nodes x 12
    xx = torch.tensor(xx, dtype=torch.float, device=device)    
    xx = torch.reshape(xx, (num_nodes, -1))
    min_val_mse = sys.float_info.max
    for i in range(args.epochs):
        model.train()        
        dataloader['train_loader'].shuffle()           
        for it, (x, y) in enumerate(tqdm(dataloader['train_loader'].get_iterator())): 
            if args.use_static_feat:
                model.embed_forward(xx)
            trainx = torch.tensor(x, device=device, dtype=torch.float)                
            trainx = trainx.transpose(1, 3)
            trainy = torch.tensor(y, device=device, dtype=torch.float)          
            output = model(trainx).squeeze()
            output = scaler.inverse_transform(output)
            curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
            curr_loss /= num_val_entry

            _optimizer.zero_grad()
            curr_loss.backward()
            _optimizer.step()

        model.eval()        
        with torch.no_grad():     
            if args.use_static_feat:
                model.embed_forward(xx)            
            val_mse = model.test_model(dataloader['val_loader'], scaler, device)
            test_mse = model.test_model(dataloader['test_loader'], scaler, device)    
            print(f'epoch: {i},  valid mse: {math.sqrt(val_mse)}, test mse: {math.sqrt(test_mse)}')
            with open(args.save_path + ".txt", "a") as f:
                f.write(f'epoch: {i},  valid mse: {math.sqrt(val_mse)}, test mse: {math.sqrt(test_mse)}\n')

            if min_val_mse > val_mse:
                min_val_mse = val_mse
                min_params = copy.deepcopy(model.state_dict())
                torch.save(min_params, args.save_path + ".pt")