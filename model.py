import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
        

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
    def __init__(self, device, num_nodes, dropout=0.3, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        self.num_nodes = num_nodes
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)        

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

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.bn.append(nn.BatchNorm2d(residual_channels))
                receptive_field += new_dilation
                new_dilation *=2
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


    def initialize(self):
        for mlist in [self.filter_convs, self.gate_convs, self.residual_convs, self.skip_convs, self.bn, self.gconv]:
            for _layer in mlist:
                _layer.reset_parameters()

        self.start_conv.reset_parameters()    
        self.nodevec1.data.copy_(torch.randn(self.num_nodes, 10))
        self.nodevec2.data.copy_(torch.randn(10, self.num_nodes))
        self.end_conv_1.reset_parameters()
        self.end_conv_2.reset_parameters()
        

    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
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
            x = self.gconv[i](x, adj)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x