import numpy as np
import torch
import os


def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, device):
    dis = torch.tensor(0.0).to(device)
    for ig in range(len(gw_real)):
        gwr = gw_real[ig]
        gws = gw_syn[ig]
        dis += distance_wb(gwr, gws)
    return dis

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def add_window_horizon(data):
    length = len(data)
    end_index = length - 23
    X = []      #windows
    Y = []      #horizon
    index = 0
    while index < end_index:
        X.append(data[index:index+12])
        Y.append(data[index+12:index+24])
        index = index + 1        

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def data_loader(X, Y, batch_size):
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)
    _data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(_data, batch_size=batch_size,
                                             shuffle=True, drop_last=False)
    return dataloader

    
def get_dataloader(args):

    #output B, N, D
    if args.dataset == 'PEMS04':
        data_path = os.path.join('../data/PEMS04/pems04.npz')
        data = np.load(data_path)['data']
    elif args.dataset == 'PEMS08':
        data_path = os.path.join('../data/PEMS08/pems08.npz')
        data = np.load(data_path)['data']
    else:
        raise ValueError

    mean, std = data.mean(), data.std()
    scaler = StandardScaler(mean, std)
    data = scaler.transform(data)
    data_len = len(data)
    
    # split data
    data_test = data[-int(data_len*0.2):]
    data_val = data[-int(data_len*0.4):-int(data_len*0.2)]
    data_train = data[:-int(data_len*0.4)]

    # build data
    x_train, y_train = add_window_horizon(data_train)
    x_val, y_val = add_window_horizon(data_val)
    x_test, y_test = add_window_horizon(data_test)
    print('Train: ', x_train.shape, y_train.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    # make data loader
    train_dataloader = data_loader(x_train, y_train, args.batch_size)
    val_dataloader = data_loader(x_val, y_val, args.batch_size)
    test_dataloader = data_loader(x_test, y_test, args.batch_size)

    return train_dataloader, val_dataloader, test_dataloader, scaler, data