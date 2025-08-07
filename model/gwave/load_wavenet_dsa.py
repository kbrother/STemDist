from tqdm import tqdm
from model.gwave.gwave import gwnet
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import argparse
import numpy as np
import util
import sys
import copy
import math


def test_gwnet(args, data, raw_data, device):
    synx = raw_data['x'].to(device)
    syny = raw_data['y'].to(device)
    _weight = raw_data['w'].to(device)
    
    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    scaler = data['scaler']
    seq_len = data['train_loader'].xs.shape[1]
    _model =  gwnet(device, num_nodes, args.dropout, in_dim, seq_len, residual_channels=args.nhid, 
                  use_model=True, dilation_channels=args.nhid, skip_channels=8*args.nhid, 
                    end_channels=16*args.nhid, ne_dim=args.ne_dim)

    _model.to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
    min_val_loss = sys.float_info.max

    nm_input_real = np.mean(data['train_loader'].xs_orig, axis=0)  # seq_length x num nodes x in_dim
    nm_input_real = np.transpose(nm_input_real, (1, 0, 2))   # num_nodes x seq length x in_dim
    nm_input_real = torch.tensor(nm_input_real, dtype=torch.float, device=device)    
    nm_input_real = torch.reshape(nm_input_real, (num_nodes, -1))   # num_nodes x seq length*in_dim
     
    nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
    nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
    nm_input_syn = torch.reshape(nm_input_syn, (synx.shape[2], -1))
    num_sample = round(synx.shape[2]/4)
    
    for i in tqdm(range(args.epochs)):
        _model.train()
        _order = list(range(synx.shape[2]))
        random.shuffle(_order)

        for j in range(4):
            if j < 3:
                _idx = _order[num_sample*j:num_sample*(j+1)]
            else:
                _idx = _order[num_sample*j:]     
            curr_weight = _weight[:,:,_idx] / torch.sum(_weight[:,:,_idx]) * num_sample
            _model.embed_forward(nm_input_syn[_idx])
        
            output_syn = _model(synx[:,:,_idx,:].transpose(1,3)).squeeze()
            loss_syn = torch.square(output_syn - syny[:,:,_idx]) * curr_weight
            loss_syn = torch.mean(loss_syn)
            
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

        _model.eval()
        if (i+1)%5 == 0:
            with torch.no_grad():
                _model.embed_forward(nm_input_real)
                val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))
    
            print(f"epoch :{i}, train loss: {loss_syn.item()} val loss: {val_loss}")
            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():
        _model.embed_forward(nm_input_real)
        test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))

    print(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")
    with open(args.save_path, "a") as f:
        f.write(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}\n")


# python -m model.gwave.load_wavenet_dsa -de 0 -d ../data/GBA -lrs 1e-2 -e 400 -lp results/dc_dsa_cluster_gba_1e-3_1e-3.pt -s 0
# python -m model.gwave.load_wavenet_dsa -de 0 -d ../data/GLA -lrs 1e-2 -e 400 -lp results/dc_dsa_cluster_gla_1e-3_1e-2_2.pt -s 0
# python -m model.gwave.load_wavenet -de 7 -d ../data/GLA -lrs 1e-3 -lp results/dc_clus_gla.pt
# python -m model.gwave.load_wavenet -de 7 -d ../data/ERA5 -lrs 1e-3 -lp results/dc_clus_era5.pt
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**5, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-lp', '--load_path', type=str, default='results/') 
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-dr', '--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('-ned', '--ne_dim',type=int,default=32,help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    raw_data = torch.load(args.load_path)

    test_gwnet(args, dataloader, raw_data, device)