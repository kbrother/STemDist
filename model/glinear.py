import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class GLinear(nn.Module):

    def __init__(self, args, num_nodes, input_dim):
        super(GLinear,self).__init__()        
        self.input_len = 12
        self.output_len = 12
        self.num_nodes = num_nodes
        self.end_conv = nn.Conv2d(self.input_len, self.output_len, (1, args.rnn_units))
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        self.linear = nn.Linear(input_dim, args.rnn_units)
        self.linear2 = nn.Linear(args.rnn_units, args.rnn_units)
    

    def forward(self, source):
        # source: B, T_1, N, dim_in
        # output: B, T_2, N

        supports = F.softmax(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1)), dim=1)
        AX = torch.einsum('nm,btmd->btnd', supports, source)   # B, T_1, N, dim_in
        AXW = torch.relu(self.linear(AX))  # B, T_1, N, h
        h = torch.einsum('nm,btmh->btnh', supports, AXW)   # B, T_1, N, h
        h = torch.relu(self.linear2(h))
        output = self.end_conv(h)  # B, T_2, N
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