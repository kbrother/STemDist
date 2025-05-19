import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        stdv = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = torch.empty(4 * hidden_size, input_size)
        self.weight_ih = Parameter(nn.init.uniform_(self.weight_ih, -stdv, stdv))
        self.weight_hh = torch.empty(4 * hidden_size, hidden_size)
        self.weight_hh =  Parameter(nn.init.uniform_(self.weight_hh, -stdv, stdv))   

        self.bias_ih = torch.empty(4 * hidden_size)
        self.bias_ih = Parameter(nn.init.uniform_(self.bias_ih, -stdv, stdv))
        self.bias_hh = torch.empty(4 * hidden_size)
        self.bias_hh = Parameter(nn.init.uniform_(self.bias_hh, -stdv, stdv))


    @jit.script_method
    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = LSTMCell(*cell_args)

    @jit.script_method
    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(list[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class NodeEmbedding_rnn(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()
        self.rnn = LSTMLayer(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, rank)
        self.hidden_size = hidden_size
        
    # X: num node x 24
    def forward(self, X):
        h0 = torch.zeros(1, self.hidden_size, device=X.device, dtype=torch.float)
        c0 = torch.zeros(1, self.hidden_size, device=X.device, dtype=torch.float)
        output, (h_n, c_n) = self.rnn(X.unsqueeze(1), (h0, c0))   #  (num node, hiddeen dim)
        return F.relu(self.linear(output.squeeze()))  #   num node x rank


class NodeEmbedding_rnn_prev(nn.Module):
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
        self.attn1 = nn.MultiheadAttention(hidden_size, 2)      
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, rank)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
    
    def forward(self, X):        
        X = F.relu(self.linear1(X))
        H, _ = self.attn1(X, X, X)
        H = self.layer_norm1(H + X)        
        output =self.linear2(H)
        return output


class NodeEmbedding_mlp(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, rank)

    def forward(self, X):
        output = F.relu(self.linear1(X))            
        return self.linear2(output)


class NodeEmbedding_mlp2(nn.Module):
    def __init__(self, input_size, rank):
        super().__init__()        
        self.linear1 = nn.Linear(input_size, rank)

    def forward(self, X):
        output = F.tanh(self.linear1(X))
        return output


class NodeEmbedding_rnn_mlp(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()               
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, rank)
        #self.rnn = nn.LSTM(input_size, hidden_size)
        self.rnn = LSTMLayer(input_size, hidden_size)

    # num series x num nodes x 12
    def forward(self, X):
        #_, (h_n, c_n) = self.rnn(X)   # h_n: 1 x num_nodes x hidden
        h0 = torch.zeros(X.shape[1], self.hidden_size, device=X.device, dtype=torch.float)
        c0 = torch.zeros(X.shape[1], self.hidden_size, device=X.device, dtype=torch.float)
        output, (h_n, c_n) = self.rnn(X, (h0, c0))   #  (num node, hidden dim)
        output = F.relu(self.linear(h_n.squeeze()))         
        return output