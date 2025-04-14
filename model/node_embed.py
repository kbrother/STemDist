import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class NodeEmbedding_rnn(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, rank)

    # X: num node x 24
    def forward(self, X):
        output, (h_n, c_n) = self.rnn(X)   #  (num node, hiddeen dim)
        return F.relu(self.linear(output))  #   num node x rank


class NodeEmbedding_birnn(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, rank)

    # X: num node x 24
    def forward(self, X):
        output, (h_n, c_n) = self.rnn(X)   #  (num node, hiddeen dim)
        return F.relu(self.linear(output))  #   num node x rank


class NodeEmbedding_attn(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, 8)        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, rank)


    def forward(self, X):
        output = F.relu(self.linear1(X))
        output, _ = self.attn(output, output, output)
        return F.relu(self.linear2(output))