import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class NodeEmbedding_rnn(nn.Module):
    def __init__(self, seq_len, hidden_size, rank):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(seq_len, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, rank)

    # X: batch size x seq len x num point
    def forward(self, X):
        X = torch.transpose(X, 1, 2)  # batch size x num point x seq len
        X, _ = self.rnn(X)   # batch size x num point x hidden dim
        return F.relu(self.linear(X))  #  batch size x num point x rank


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


class NodeEmbedding2(nn.Module):
    def __init__(self, num_node, rank):
        super().__init__()
        self.embed = nn.Parameter(torch.rand(num_node, rank))
        
    # X: batch size x seq len x num point
    def forward(self):
        return self.embed


class NodeEmbedding3(nn.Module):
    def __init__(self, input_dim, hidden_size, rank):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(input_dim, hidden_size, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_size, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, rank)

    # X: num point x 20
    def forward(self, X):
        Q = self.W_q(X)  # num point x hidden dim
        K = self.W_k(X)  # num point x hidden dim
        V = self.W_v(X)  # num point x hidden dim
        E = F.softmax(torch.mm(Q, torch.transpose(K, 0, 1))/math.sqrt(self.hidden_size))  # num point x num point
        E = torch.mm(E, V)  #  num point x hidden dim
        return F.relu(self.linear(E))  #  num point x rank