import torch.nn as nn
import torch
import copy
import sys
import util
import random
from model.gwave import gwnet
from tqdm import tqdm
import numpy as np
import argparse


class Coreset:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate *  data['train_loader'].xs.shape[0])


    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx
        syny = self.syny

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']
        _model = gwnet(self.device, num_nodes, 0.3, in_dim, 12, 
                           residual_channels=args.nhid, dilation_channels=args.nhid, 
                           skip_channels=8*args.nhid, end_channels=16*args.nhid)
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.learning_rate)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(300)):
            _model.train()
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            output_syn = scaler.inverse_transform(output_syn)
            loss_syn, num_val_entry = util.masked_se(output_syn, syny, 0.)
            loss_syn /= num_val_entry
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():
                    val_loss = _model.test_model(data['val_loader'], scaler)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

                print(f'min i: {min_i}, val: {val_loss}')
                
        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            test_loss = _model.test_model(data['test_loader'], scaler)

        return min_i, min_val_loss, test_loss


class RandomSample(Coreset):
    def __init__(self, data, args, device):
        super().__init__(data, args, device)
        
        num_total = data['train_loader'].xs.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = self.data['train_loader'].xs[sampled_idx]
        self.syny = self.data['train_loader'].ys[sampled_idx, :, :, 0]
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
        print(f'x: {self.synx.shape}, y:{self.syny.shape}')
    
    def train(self):
        min_i, val_loss, test_loss = self.test_syn()
        print(f"min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        with open(self.args.save_path, 'a') as f:
            f.write(f"min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        


# python -m Sampling.core_set_gwave -de 0 -s 0 -lr 1e-2 -r 1e-3
# python -m Sampling.core_set_gwave -d ../data/PEMS-BAY -de 4 -s 0 -lr 1e-2 -r 3e-4
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-r', '--reduction_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=2**11, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")

    _model = RandomSample(dataloader, args, device)
    min_i, min_val_loss, test_loss = _model.test_syn()
    print(f"min i: {min_i}, min val loss: {min_val_loss}, test loss: {test_loss}")
    