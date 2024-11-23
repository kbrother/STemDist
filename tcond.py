import torch.nn as nn
from tqdm import tqdm
from model import gwnet


class TCond:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate * data['train_loader'].num_elems)

        # Define condensed data
        _shape = list(data['train_loader'].xs.shape[1:])
        _shape = tuple([self.num_elems] + _shape)
        self.feat_x = torch.rand(_shape, device=device, dtype=torch.float)
        self.feat_x = nn.Parameter(self.feat_x)

        _shape = list(data['train_loader'].ys.shape[1:])
        _shape = tuple([self.num_elems] + _shape)
        self.feat_y = torch.rand(_shape, device=device, dtype=torch.float)
        self.feat_y = nn.Parameter(self.feat_y)
        print(f'feat x shape: {self.feat_x.shape}')
        print(f'feat y shape: {self.feat_y.shape}')

    
    def train(self):
        args = self.args
        data = self.data
        feat_x, feat_y = self.feat_x, self.feat_y

        for i in tqdm(range(args.epochs)):
            _model = gwnet(device, num_nodes, args.dropout, in_dim, args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=8*args.nhid, end_channels=16*args.nhid)