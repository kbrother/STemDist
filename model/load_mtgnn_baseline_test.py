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
import time

def test_syn(args, synx, syny, device):    
    num_nodes = synx.shape[2]
    in_dim = synx.shape[3]    
    seq_len = synx.shape[1]    
    out_dim = 1
    _model = gtnet(True, True, 2, num_nodes, 
                  device, predefined_A=None, use_static_feat=False,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)      
    _model.to(device)
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=0.01)
    _model = _model.to(device)
   
    min_val_loss = sys.float_info.max
    for i in range(15):
        start_time = time.time()
        _model.train()
        output_syn = _model(synx.transpose(1, 3)).squeeze()
        loss_syn = F.mse_loss(output_syn, syny)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

        if i >=5: 
            print(time.time() - start_time)

    print(f"memory: {torch.cuda.memory.memory_reserved(device)/10**9}")
    

# python -m model.load_mtgnn_baseline_test -de 1 -lp results/dc_gba_1e-2_1e-3_1.pt -s 0
# python -m model.load_mtgnn_baseline_test -de 1 -lp results/dc_gla_1e-2_1e-3_1.pt -s 0
# python -m model.load_mtgnn_baseline_test -de 1 -lp results/dc_era5_1e-2_1e-3_1.pt -s 0
# python -m model.load_mtgnn_baseline_test -de 1 -lp results/dc_ca_1e-2_1e-2_1.pt -s 0
# python -m model.load_mtgnn_baseline_test -de 1 -lp results/dc_aurora_1e-2_1e-4_1.pt -s 0
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-lp', '--load_path', type=str, default='results/') 
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")

    raw_data = torch.load(args.load_path)
    synx = raw_data['x'].to(device)
    syny = raw_data['y'].to(device)
    
    if (len(synx.shape) <=3):
        synx = synx.unsqueeze(-1)
        
    print("load finish")
    print(synx.shape)
    print(syny.shape)
    test_syn(args, synx, syny, device)