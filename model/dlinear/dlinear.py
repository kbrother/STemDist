from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import util

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, seq_len, pred_len, channels):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)        
        self.channels = channels
        
        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
   
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]


    def test_model(self, dataloader, scaler, device):
        loss_sum, naive_loss_sum = 0, 0

        y_mean = 0
        num_entry = 0
        for iter, (x, y) in enumerate(dataloader.get_iterator()):
            valy = torch.tensor(y, device=device, dtype=torch.float)
            mask = (valy != 0.).float()            
            num_entry += torch.sum(mask)
            y_mean += torch.sum(valy)
        y_mean /= num_entry
            
        for iter, (x, y) in enumerate(dataloader.get_iterator()):
            valx = torch.tensor(x, device=device, dtype=torch.float)
            valx = valx[..., 0]
            valy = torch.tensor(y, device=device, dtype=torch.float)
            output = self.forward(valx).squeeze()
            output = scaler.inverse_transform(output)
            #valy = scaler.transform(valy)
            curr_loss, curr_naive_loss = util.masked_se2(output, valy, 0., y_mean)
            loss_sum += curr_loss.item()
            naive_loss_sum += curr_naive_loss.item()
        return loss_sum/naive_loss_sum