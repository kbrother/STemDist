import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class GRNN_complex(nn.Module):
    def __init__(self, args, num_nodes, input_dim, node_embed=None):
        super(GRNN,self).__init__()
        self.input_len = 12
        self.output_len = 12
        self.embed_dim = 10
        self.num_nodes = num_nodes
        self.rnn_units = args.rnn_units
        if node_embed is None:
            self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim))
        else:
            self.node_embeddings = nn.Parameter(node_embed)
        self.weights = nn.Parameter(torch.randn(self.embed_dim, input_dim+args.rnn_units, args.rnn_units))
        self.bias = nn.Parameter(torch.randn(self.embed_dim, args.rnn_units))        
        self.end_conv = nn.Conv2d(1, self.output_len, (1, args.rnn_units))

    
    def forward(self, source):
        # source: B, T_1, N, 2
        # output: B, T_2, N
        adj = F.softmax(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1)), dim=1)
        weights = torch.einsum('nd,dio->nio', self.node_embeddings, self.weights)  # N, 2+R, R
        bias = torch.mm(self.node_embeddings, self.bias)  # N, R

        seq_len = source.shape[1]
        batch_size = source.shape[0]
        prev_state = torch.zeros(batch_size, self.num_nodes, self.rnn_units).to(source.device)        # B, N, R
        for i in range(seq_len):
            X = torch.cat((source[:,i,:,:], prev_state), dim=-1)  # B, N, R+2
            AX = torch.einsum("nm,bmc->bnc", adj, X)   # B, N, R+2
            AXW = torch.einsum("bnc,ncr->bnr", AX, weights) + bias.unsqueeze(0)  # B, N, R
            prev_state = torch.relu(AXW)   # B, N, R

        return self.end_conv(prev_state.unsqueeze(1)).squeeze()


class GRNN(nn.Module):
    def __init__(self, args, num_nodes, input_dim, node_embed=None):
        super(GRNN,self).__init__()
        self.input_len = 12
        self.output_len = 12
        self.embed_dim = 10
        self.num_nodes = num_nodes
        self.rnn_units = args.rnn_units
        if node_embed is None:
            self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim))
        else:
            self.node_embeddings = nn.Parameter(node_embed)

        self.linear = nn.Linear(input_dim+args.rnn_units, args.rnn_units)  
        self.end_conv = nn.Conv2d(1, self.output_len, (1, args.rnn_units))

    
    def forward(self, source):
        # source: B, T_1, N, 2
        # output: B, T_2, N
        adj = F.softmax(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1)), dim=1)
        seq_len = source.shape[1]
        batch_size = source.shape[0]
        prev_state = torch.zeros(batch_size, self.num_nodes, self.rnn_units).to(source.device)        # B, N, R
        for i in range(seq_len):
            X = torch.cat((source[:,i,:,:], prev_state), dim=-1)  # B, N, R+2
            AX = torch.einsum("nm,bmc->bnc", adj, X)   # B, N, R+2
            AXW = self.linear(AX)  # B, N, R
            prev_state = torch.relu(AXW)   # B, N, R

        return self.end_conv(prev_state.unsqueeze(1)).squeeze()

    
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