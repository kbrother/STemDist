import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class NodeEmbedding_rnn(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, rank)

    # X: num time x 12 x node idx x 2
    def forward(self, X):
        X = torch.transpose(X, 1, 2)  # num time x node idx x 12 x 2
        num_time, num_nodes = X.shape[0], X.shape[1]
        X = X.reshape(num_time, num_nodes, -1)   # num_time x node idx x 24
        _, (h_n, c_n) = self.rnn(X)   # (1, num node, hiddeen dim)
        return F.relu(self.linear(h_n.squeeze()))  #   num node x rank


class NodeEmbedding_node_rnn(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, rank)

    # X: num node x 24
    def forward(self, X):
        output, (h_n, c_n) = self.rnn(X)   #  (num node, hiddeen dim)
        return F.relu(self.linear(output))  #   num node x rank


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = torch.cat((x, self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)), 2)
        return x  # (batch_size, seq_len, 2*d_model)


class NodeEmbedding_attn(nn.Module):

    def __init__(self, max_seq, input_size, hidden_size, rank):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, rank)
        self.linear_attn = nn.Linear(2*hidden_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, max_seq)
        self.attn_w = nn.Parameter(torch.randn(hidden_size))
    
    # X: num time x 12 x node idx x 2
    def forward(self, X):
        X = torch.permute(X, (2, 0, 1, 3))  # node idx x num time x 12 x 2
        num_time, num_nodes = X.shape[1], X.shape[0]
        X = X.reshape(num_nodes, num_time, -1)   # node idx x num time x 24
        X = F.relu(self.linear1(X)) # node idx x num time x hidden

        attn_weight = F.tanh(self.linear_attn(self.pos_enc(X)))   # node idx x num time x hidden
        attn_weight = torch.sum(attn_weight * self.attn_w.unsqueeze(0).unsqueeze(0), dim=-1)  # node idx x num time
        attn_weight = F.softmax(attn_weight, dim=-1)   # node idx x num time
        X = X * attn_weight.unsqueeze(-1)  # node idx x num time x hidden
        X = torch.sum(X, dim=1) #  node idx x hidden
        return F.relu(self.linear2(X))  # node idx x rank
        
