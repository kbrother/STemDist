import numpy as np
import torch
import os


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