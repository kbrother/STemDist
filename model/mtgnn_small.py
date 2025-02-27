from model.layers_small import *
import util


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, device, predefined_A=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=16, residual_channels=16, skip_channels=32, end_channels=64, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        if not gcn_true:
            self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(subgraph_size, node_dim, device, alpha=tanhalpha)

        self.seq_length = seq_length
        kernel_size = 3
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                if not gcn_true:
                    self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                         out_channels=residual_channels,
                                                         kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))              

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, 207, self.seq_length - rf_size_j + 1), elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, 207, self.receptive_field - rf_size_j + 1), elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(207).to(device)


    def set_node_embed(self, node_embed):
        self.gc.register_buffer("node_embed", node_embed)
        
        
    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc()
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            x = torch.tanh(filter)            
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            # B x C X N x L'

            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

            '''
            num_nodes = x.shape[2]
            num_channel = x.shape[1]
            num_l = x.shape[3]
            x = torch.transpose(x, 1, 2).reshape(-1, num_channel, num_l) #BN X C X L'
            x = self.norm[i](x)
            x = x.reshape(-1, num_nodes, num_channel, num_l) #B x N X C X L'
            x = torch.transpose(x, 1, 2) 
            ''' 
        
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

    
    def test_model(self, embedding, dataloader, scaler, device):
        loss_sum, num_entry = 0, 0
        for iter, (x, y) in enumerate(dataloader.get_iterator()):        
            valx = torch.tensor(x, device=device, dtype=torch.float)
            node_embed = embedding(valx[...,0])
            node_embed = torch.mean(node_embed, dim=0)
            self.set_node_embed(node_embed)
            
            valx = valx.transpose(1, 3)
            valy = torch.tensor(y, device=device, dtype=torch.float)
            valy = valy[:,:,:,0]
            output = self.forward(valx).squeeze()
            output = scaler.inverse_transform(output)
            curr_loss, num_curr_entry = util.masked_se(output, valy, 0.)
            loss_sum += curr_loss.item()
            num_entry += num_curr_entry.item()               
        return loss_sum/num_entry