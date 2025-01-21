import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class Linear(nn.Module):

    def __init__(self, args, input_dim):
        super(GLinear,self).__init__()        
        self.input_len = 12
        self.output_len = 12
        self.end_conv = nn.Conv2d(self.input_len, self.output_len, (1, input_dim))
        

    def forward(self, source):
        # source: B, T_1, N, dim_in
        # output: B, T_2, N
        output = self.end_conv(source)  # B, T_2, N
        return output.squeeze()

    
    def test_model(self, dataloader, scaler, device):
        loss_sum, num_entry = 0, 0
        for iter, (x, y) in enumerate(dataloader.get_iterator()):
            valx = torch.tensor(x, device=device, dtype=torch.float)
            valy = torch.tensor(y, device=device, dtype=torch.float)
            valy = valy[:,:,:,0]
            output = self.forward(valx).squeeze()
            output = scaler.inverse_transform(output)
            curr_loss, num_curr_entry = util.masked_se(output, valy, 0.)
            loss_sum += curr_loss.item()
            num_entry += num_curr_entry.item()               
        return loss_sum/num_entry