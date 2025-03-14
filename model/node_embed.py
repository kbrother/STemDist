import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class NodeEmbedding(nn.Module):
    def __init__(self, seq_len, hidden_size, rank):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(seq_len, hidden_size, bias=False)
        self.W_k = nn.Linear(seq_len, hidden_size, bias=False)
        self.W_v = nn.Linear(seq_len, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, rank)

    # X: batch size x seq len x num point
    def forward(self, X):
        X = torch.transpose(X, 1, 2)
        Q = self.W_q(X)  # batch size x num point x hidden dim
        K = self.W_k(X)
        V = self.W_v(X)
        E = F.softmax(torch.bmm(Q, torch.transpose(K, 1, 2))/math.sqrt(self.hidden_size), dim=-1)  # batch size x num point x num point
        E = torch.bmm(E, V)  #  batch size x num point x hidden dim
        return F.relu(self.linear(E))  #  batch size x num point x rank
        