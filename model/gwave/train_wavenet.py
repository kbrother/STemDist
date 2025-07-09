import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.gwave.gwave import gwnet
import torch.optim as optim
import math
from scipy.io import loadmat
import sys
import numpy as np


# python -m model.gwave.train_wavenet -de 7 -d ../data/GBA -lr 1e-2 -e 100 -sp results/real_gwave_gba.txt -b 32
# python -m model.gwave.train_wavenet -de 5 -d ../data/GLA -lr 1e-3 -e 100 -sp results/real_gwave_gla.txt -b 32
# python -m model.gwave.train_wavenet -de 5 -d ../data/ERA5 -lr 1e-3 -e 100 -sp results/real_gwave_era5.txt -b 32
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
    parser.add_argument('-sp', '--save_path', type=str, default='results/', help='data path')
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
    model = gwnet(device, num_nodes, args.dropout, in_dim, args.seq_length, residual_channels=args.nhid,                   
                  dilation_channels=args.nhid, skip_channels=8*args.nhid, end_channels=16*args.nhid)
    model.to(device)
    #model.reset_parameters()
    
    _optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    min_val_mse = sys.float_info.max
    for i in range(args.epochs):

        model.train()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
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
            val_mae = math.sqrt(model.test_model(dataloader['val_loader'], scaler, device))
            test_mae = math.sqrt(model.test_model(dataloader['test_loader'], scaler, device))
            print(f'epoch: {i}, valid mae: {val_mae}, test mae: {test_mae}')

            with open(args.save_path, "a") as f:
                f.write(f'epoch: {i}, valid mae: {val_mae}, test mae: {test_mae}\n')