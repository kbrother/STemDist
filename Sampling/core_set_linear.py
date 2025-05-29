import torch.nn as nn
import torch
import copy
import sys
import util
import random
from model.linear import Linear
from tqdm import tqdm
import numpy as np
import argparse
import torch.nn.functional as F


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
        input_len = data['train_loader'].xs.shape[1]
        output_len = input_len
        _model = Linear(args, input_len, output_len, in_dim)
        _model.to(self.device)
        
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.learning_rate)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            _model.train()
            output_syn = _model(synx)
            output_syn = scaler.inverse_transform(output_syn)
            #loss_syn, num_val_entry = util.masked_se(output_syn, syny, 0.)
            #loss_syn /= num_val_entry
            loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():
                    val_loss = _model.test_model(data['val_loader'], scaler, self.device)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())
                print(f'epoch: {i}, val loss: {val_loss}')

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            test_loss = _model.test_model(data['test_loader'], scaler, self.device)

        return min_i, min_val_loss, test_loss


class RandomSample(Coreset):
    def __init__(self, data, args, device):
        super().__init__(data, args, device)
        
        num_total = data['train_loader'].xs.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = self.data['train_loader'].xs[sampled_idx]
        self.syny = self.data['train_loader'].ys[sampled_idx, :, :]
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
        print(f'x: {self.synx.shape}, y:{self.syny.shape}')


# python -m Sampling.core_set_linear -d ../data/METR-LA -de 1 -s 0 -lr 1e-1 -r 2e-3
# python -m Sampling.core_set_linear -d ../data/PEMS-BAY -de 1 -s 0 -lr 1e-1 -r 2e-3
# python -m Sampling.core_set_linear -d ../data/ELECTRICITY -de 1 -s 0 -lr 1e-1 -r 2e-3
# python -m Sampling.core_set_linear -d ../data/SOLAR -de 1 -s 0 -lr 1e-1 -r 2e-3
# python -m Sampling.core_set_linear -d ../data/TRAFFIC -de 1 -s 0 -lr 1 -r 2e-3
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA-Tensor', help='data path')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-r', '--reduction_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
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
    