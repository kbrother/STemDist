import torch
import numpy as np
import argparse
# from model.dlinear.dlinear import Model
# from model.node_embed import *
import util
from tqdm import tqdm
import random
from model import *
import math
import sys
from TimeDC.reparam_module import ReparamModule
import copy
from TimeDC.network_patch import TSFE_Model
import torch.nn as nn
import torch.nn.functional as F




if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch sizefor real data')#128
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate for testing on synthetic data')
    parser.add_argument('-rr', '--series_reduce_rate',type=float,default=2e-2,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/') 
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    parser.add_argument('-ne', '--num_experts', type=int, default=20, help='')

    parser.add_argument('--params', type=str, default='../data/params/METR-LA-MTGNN/')
    parser.add_argument('--max_start_epoch', type=int, default=4, help='max epoch we can start at')
    parser.add_argument('--expert_epoch', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=10, help='how many steps to take on synthetic data')
    parser.add_argument('--beta', type=float, default=0.01, help='CondTSF addtive ratio')

    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=1, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader =  util.load_dataset(args.data, args.batch_size)
    print("load finish")
    
    scaler = dataloader['scaler']
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]
    if len(dataloader['train_loader'].ys.shape) == 4:        
        out_dim = dataloader['train_loader'].ys.shape[3]
    else: 
        out_dim = 1
    seq_len = dataloader['train_loader'].xs.shape[1]

    for it in range(args.num_experts):
        model = TSFE_Model(args, num_features=in_dim).to(args.device)
        model.to(device)
        #model.reset_parameters()

        curr_traj = [[p.detach().cpu() for p in model.parameters()]]        
        _optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        
        for i in tqdm(range(args.epochs)):
            model.train()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.tensor(x, device=device, dtype=torch.float)
                trainx = trainx[:,:,:,0]#.squeeze(3)
                # print('trainx: ', trainx)
                trainy = torch.tensor(y, device=device, dtype=torch.float)
                if len(trainy.shape) == 4:
                    trainy = trainy[:,:,:,0]
                output, _, _ = model(trainx)
                if torch.isnan(output).any():
                    print('nan in output')
                    print('output: ', output.shape)
                output = scaler.inverse_transform(output)
                curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
                curr_loss /= num_val_entry
    
                _optimizer.zero_grad()
                curr_loss.backward()
                _optimizer.step()
    
            model.eval()
            with torch.no_grad():               
                val_mae = model.test_model(dataloader['val_loader'], scaler, device)
                test_mae = model.test_model(dataloader['test_loader'], scaler, device)            
                print(f'epoch: {i}, valid mae: {math.sqrt(val_mae)}, test mae: {math.sqrt(test_mae)}')

            curr_traj.append([p.detach().cpu() for p in model.parameters()])        
        torch.save(curr_traj, args.save_path + f"replay_buffer_{5*args.seed + it}.pt")
        

# python -m TimeDC.buffer_timedc -de 0 -lr 1e-5 -e 40 -s 0 -ne 5 -d ../data/ERA5 -sp ../data/params/ERA5-TimeDC/