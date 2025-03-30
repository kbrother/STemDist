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
        x = x + self.pe[:, :x.size(1)]
        return x


batch_size = 32
seq_len = 100
d_model = 512

x = torch.zeros(batch_size, seq_len, d_model)  # your input embeddings
pos_encoder = PositionalEncoding(d_model)
x = pos_encoder(x)

class NodeEmbedding_attn(nn.Module):

    def __init__(self, input_size, hidden_size, rank):
        super().__init__()

        self.linear1 = nn.Linear(hidden_size, input_size)
        