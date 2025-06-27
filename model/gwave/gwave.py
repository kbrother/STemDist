import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import util
from model.node_embed import *


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()
        

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

    def reset_parameters(self):
        self.mlp.reset_parameters()
        

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order + 1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, a):
        out = [x]        
        x1 = self.nconv(x,a)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1,a)
            out.append(x2)
            x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

    def reset_parameters(self):
        self.mlp.reset_parameters()
        

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, blocks=4,layers=2, use_model=False, ne_dim=128):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        self.device = device
        self.num_nodes = num_nodes

        if use_model:
            self.model1 = NodeEmbedding_attn(out_dim*in_dim, ne_dim, 10)
            self.model2 = NodeEmbedding_attn(out_dim*in_dim, ne_dim, 10)
        else:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)        
            #self.nodevec2 = self.nodevec1.T
        
        for b in range(blocks):
            new_dilation = 1       
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, 2),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, 2), dilation=new_dilation))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.bn.append(nn.BatchNorm2d(residual_channels))
                receptive_field += new_dilation
                new_dilation *=2
                if (b<blocks-1) or (i<layers - 1):
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        

    def embed_forward(self, _input):
        node_embed1 = self.model1(_input)
        node_embed2 = self.model2(_input)
        self.set_node_embed(node_embed1, node_embed2)

    
    def set_node_embed(self, node_embed1, node_embed2):
        self.register_buffer("nodevec1", node_embed1)
        self.register_buffer("nodevec2", node_embed2)

    
    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2.t())), dim=1)
        #adj = F.relu(F.tanh(torch.mm(self.nodevec1, self.nodevec2)))
        for i in range(self.blocks * self.layers):
            # Gated TCN
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # Skip connection for the final output
            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = skip + s

            # GCN layer
            if i<(self.blocks*self.layers)-1:
                x = self.gconv[i](x, adj)
                x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


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
            valx = valx.transpose(1, 3)
            valy = torch.tensor(y, device=device, dtype=torch.float)
            output = self.forward(valx).squeeze()            
            output = scaler.inverse_transform(output)
            curr_loss, curr_naive_loss = util.masked_se2(output, valy, 0., y_mean)
            loss_sum += curr_loss.item()
            naive_loss_sum += curr_naive_loss.item()
        return loss_sum/naive_loss_sum
