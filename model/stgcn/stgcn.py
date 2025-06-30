import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.node_embed import *
import util


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        #self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        #return self.batch_norm(t3)
        return t3
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_features, num_timesteps_input,
                 num_timesteps_output, ne_dim=32):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

        self.model1 = NodeEmbedding_attn(num_timesteps_input*num_features, ne_dim, 10)
        self.model2 = NodeEmbedding_attn(num_timesteps_input*num_features, ne_dim, 10)

        self.node_lin1 = nn.Linear(10, 10)
        self.node_lin2 = nn.Linear(10, 10)

    
    def embed_forward(self, _input):
        node_embed1 = self.model1(_input)
        node_embed2 = self.model2(_input)
        self.register_buffer("nodevec1", node_embed1)
        self.register_buffer("nodevec2", node_embed2)
        
    
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        alpha = 3
        A_hat = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2.t())), dim=1)
        
        #mask = torch.zeros(num_node, num_node).to(X.device)
        #mask.fill_(float('0'))
        #_k = min(self.k, adj.shape[0])
        #_k = round(adj.shape[0]*0.1)
        #s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(_k,1)
        #mask.scatter_(1,t1,s1.fill_(1))
        
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4

    def test_model(self, dataloader, scaler, device):
        loss_sum, naive_loss_sum = 0, 0

        y_mean = 0
        num_entry = 0
        for iter, (x, y) in enumerate(dataloader.get_iterator()):
            valy = torch.tensor(y, device=device, dtype=torch.float)
            mask = (valy != 0.).float()            
            num_entry += torch.sum(mask)
            y_mean += torch.sum(valy)
        y_mean /= num_entry
            
        for iter, (x, y) in enumerate(dataloader.get_iterator()):
            valx = torch.tensor(x, device=device, dtype=torch.float)
            valx = valx.transpose(1, 2)
            valy = torch.tensor(y, device=device, dtype=torch.float)
            output = self.forward(valx).squeeze()            
            output = output.transpose(1,2)
            output = scaler.inverse_transform(output)
            curr_loss, curr_naive_loss = util.masked_se2(output, valy, 0., y_mean)
            loss_sum += curr_loss.item()
            naive_loss_sum += curr_naive_loss.item()
        return loss_sum/naive_loss_sum