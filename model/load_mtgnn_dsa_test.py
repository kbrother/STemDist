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
import torch.nn.functional as F


def test_syn(args, raw_data, device):
    synx = raw_data['x'].to(device)
    syny = raw_data['y'].to(device)
    if 'w' in raw_data:
        _weight = raw_data['w'].to(device)
    else:
        _weight = None
    print("load finish")
    print(synx.shape)
    print(syny.shape)
    
    num_nodes = synx.shape[2]
    in_dim = synx.shape[3]    
    seq_len = synx.shape[1]    
    out_dim = 1
    _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=args.ne_dim)      
    _model.to(device)
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=1e-3)
    min_val_loss = sys.float_info.max
    
    nm_input_syn = torch.mean(synx, dim=0)   # seq_length x num_nodes x in_dim
    nm_input_syn = torch.transpose(nm_input_syn, 0, 1)  # num nodex x seq_length x in_dim
    nm_input_syn = torch.reshape(nm_input_syn, (synx.shape[2], -1))
    
    num_sample = round(synx.shape[2]/4)
    for i in range(15):
        start_time = time.time()
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
            output_syn = _model(synx[:,:,_idx,:].transpose(1, 3)).squeeze()
            loss_syn = torch.square(output_syn - syny[:,:,_idx]) * curr_weight
            loss_syn = torch.mean(loss_syn)
                #loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()
            
        if i >=5: 
            print(time.time() - start_time)
    print(f"memory: {torch.cuda.memory.memory_reserved(device)/10**9}")

    
# python -m model.load_mtgnn_dsa_test -de 1 -lp results/dc_dsa_cluster_gba_1e-3_1e-3_1.pt -s 0
# python -m model.load_mtgnn_dsa_test -de 1 -lp results/dc_dsa_cluster_gla_1e-3_1e-2_2.pt -s 0
# python -m model.load_mtgnn_dsa_test -de 1 -lp results/dc_dsa_cluster_era5_1e-3_1e-3_1.pt -s 0
# python -m model.load_mtgnn_dsa_test -de 1 -lp results/dc_dsa_cluster_ca_1e-3_1e-3_1.pt -s 0
# python -m model.load_mtgnn_dsa_test -de 1 -lp results/dc_dsa_cluster_aurora_1.pt -s 0
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-lp', '--load_path', type=str, default='results/')
    parser.add_argument('-ned', '--ne_dim',type=int,default=32,help='')
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    raw_data = torch.load(args.load_path, map_location=device)
    test_syn(args, raw_data, device)