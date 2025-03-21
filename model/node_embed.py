import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class NodeEmbedding_rnn(nn.Module):
    def __init__(self, seq_len, hidden_size, rank):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(seq_len, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, rank)

    # X: batch size x seq len x num point
    def forward(self, X):
        X = torch.transpose(X, 1, 2)  # batch size x num point x seq len
        X, _ = self.rnn(X)   # batch size x num point x hidden dim
        return F.relu(self.linear(X))  #  batch size x num point x rank



class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        _, seq_len, _ = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512] 


class NodeEmbedding_tf_encoder(nn.Module):
    def __init__(self, seq_len, hidden_size, rank, num_nodes, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(seq_len, hidden_size)
        self.linear2 = nn.Linear(hidden_size, rank)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
                                 
    # X: batch size x seq len x num point
    def forward(self, X):
        X = torch.transpose(X, 1, 2)  # batch size x num point x seq len
        X = F.relu(self.linear1(X))   # batch size x num point x hidden size
        #X = X + self.pos_embed(X).unsqueeze(0)
        X = self.encoder(X)   # batch size x num point x hidden size
        return F.relu(self.linear2(X))  #  batch size x num point x rank


class NodeEmbedding_self_attn(nn.Module):
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