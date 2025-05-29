import torch
import torch.nn as nn
import util
import math


class Linear(nn.Module):

    def __init__(self, args, input_len, output_len, input_dim):
        super(Linear,self).__init__()        
        self.input_len = input_len
        self.output_len = output_len
        self.end_conv = nn.Conv2d(self.input_len, self.output_len, (1, input_dim))
        

    def forward(self, source):
        # source: B, T_1, N, dim_in
        # output: B, T_2, N
        output = self.end_conv(source)  # B, T_2, N
        return output.squeeze()

    
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
            valy = torch.tensor(y, device=device, dtype=torch.float)
            output = self.forward(valx).squeeze()
            output = scaler.inverse_transform(output)
            curr_loss, curr_naive_loss = util.masked_se2(output, valy, 0., y_mean)
            loss_sum += curr_loss.item()
            naive_loss_sum += curr_naive_loss.item()           
        return math.sqrt(loss_sum/naive_loss_sum)