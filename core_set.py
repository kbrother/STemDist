import torch.nn as nn
import torch
import copy
import sys
import util
import random
from model import gwnet
from tqdm import tqdm


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
        _model = gwnet(self.device, num_nodes, args.dropout, in_dim, args.seq_length, 
                           residual_channels=args.nhid, dilation_channels=args.nhid, 
                           skip_channels=8*args.nhid, end_channels=16*args.nhid)
        _model.to(self.device)
        _model = nn.DataParallel(_model, device_ids=[1,2,3])
        optimizer = torch.optim.Adam(_model.module.parameters(), lr=0.001, weight_decay=0.0001)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            _model.module.train()
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            output_syn = scaler.inverse_transform(output_syn)
            loss_syn = util.mae(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.module.eval()
            if (i+1)%20 == 0:
                with torch.no_grad():
                    val_loss = _model.module.test_model(data['val_loader'], scaler)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.module.state_dict())

        _model.module.load_state_dict(min_params)
        _model.module.eval()
        with torch.no_grad():
            test_loss = _model.module.test_model(data['test_loader'], scaler)

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
    

    def train(self):
        val_loss, test_loss = self.test_syn()
        print(f" val loss: {val_loss}, test loss: {test_loss}")
        with open(self.args.save_path, 'a') as f:
            f.write(f"val loss: {val_loss}, test loss: {test_loss}\n")        