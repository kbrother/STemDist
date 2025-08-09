import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.stgcn.stgcn import STGCN
import torch.optim as optim
import math
from scipy.io import loadmat
import sys
import numpy as np
import copy


# python -m model.stgcn.train_stgcn -de 6 -d ../data/GBA -lr 1e-2 -e 100 -sp results/real_stgcn_gba.txt
# python -m model.stgcn.train_stgcn -de 6 -d ../data/GLA -lr 1e-2 -e 100 -sp results/real_stgcn_gla.txt
# python -m model.stgcn.train_stgcn -de 6 -d ../data/ERA5 -lr 1e-2 -e 100 -sp results/real_stgcn_era5.txt
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-sp', '--save_path', type=str, default='results/', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**5, help='batch size')
    parser.add_argument('-lr', '--lr',type=float,default=1e-2,help='learning rate')    
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-e', '--epochs', default=100, type=int)
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
    _model = STGCN(in_dim, seq_len, seq_len, num_nodes, False)
    #model.reset_parameters()

    _model.to(device)
    _optimizer = optim.Adam(_model.parameters(), lr=args.lr)
    min_val_mse = sys.float_info.max    
    for i in range(args.epochs):
        _model.train()        
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.tensor(x, device=device, dtype=torch.float)
            trainx = trainx.transpose(1, 2)
            trainy = torch.tensor(y, device=device, dtype=torch.float)
            output = _model(trainx).transpose(1, 2)
            output = scaler.inverse_transform(output)
            curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
            curr_loss /= num_val_entry

            _optimizer.zero_grad()
            curr_loss.backward()
            _optimizer.step()

        _model.eval()
        with torch.no_grad():               
            val_mse = math.sqrt(_model.test_model(dataloader['val_loader'], scaler, device))
            #test_mae = math.sqrt(_model.test_model(dataloader['test_loader'], scaler, device))            
            print(f'epoch: {i}, valid mae: {val_mse}')
            if min_val_mse > val_mse:
                min_val_mse = val_mse
                min_params = copy.deepcopy(_model.state_dict())

    _model.load_state_dict(min_params)
    with torch.no_grad():
        test_mse = math.sqrt(_model.test_model(dataloader['test_loader'], scaler, device))
    with open(args.save_path, "a") as f:
        f.write(f'epoch: {i}, valid mse: {min_val_mse}, test mse: {test_mse}\n')