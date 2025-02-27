import torch.nn as nn
from tqdm import tqdm
from model.mtgnn_small import gtnet
import torch
import util
import sys
import copy
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import copy
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

    
class GradMatch:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.series_reduce_rate *  data['train_loader'].xs.shape[0])
        self.num_nodes = int(args.node_reduce_rate * data['train_loader'].xs.shape[2])
        scaler = data['scaler']
        
        # Define condensed data
        '''
        _shape = [self.num_elems] + list(data['train_loader'].xs.shape[1:])
        _shape[2] = self.num_nodes
        self.synx = torch.rand(tuple(_shape), device=device, dtype=torch.float)
        # num elems x seq len x num point x feature
        _shape = [self.num_elems] + list(data['train_loader'].ys.shape[1:-1])
        _shape[2] = self.num_nodes
        self.syny = torch.rand(_shape, device=device, dtype=torch.float)
        # num elems x seq len x num point
        '''
        # Define condensed data
        num_total = data['train_loader'].xs.shape[0]
        sampled_idx = random.sample(list(range(num_total)), self.num_elems)
        self.synx = self.data['train_loader'].xs[sampled_idx]     
        self.synx = torch.tensor(self.synx, device=device, dtype=torch.float)
        
        self.syny = self.data['train_loader'].ys[sampled_idx, :, :, 0]
        self.syny = scaler.transform(self.syny)
        self.syny = torch.tensor(self.syny, device=device, dtype=torch.float)
        
        self.synx = nn.Parameter(self.synx)
        self.syny = nn.Parameter(self.syny)
        print(f'feat x shape: {self.synx.shape}')
        print(f'feat y shape: {self.syny.shape}')
        seq_len = self.syny.shape[1]        
        self.embedding = NodeEmbedding(seq_len, 256, 10).to(self.device)
        '''
        static_feat = np.load("model/node_embed.npy", allow_pickle=True)
        #print(static_feat)
        static_feat1 = static_feat[()]['v1']
        static_feat2 = static_feat[()]['v2']
        self.static_feat1 = torch.tensor(static_feat1, device=device, dtype=torch.float)
        self.static_feat2 = torch.tensor(static_feat2, device=device, dtype=torch.float).transpose(0, 1)
        '''
        
        
    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx.detach().clone()
        syny = self.syny.detach().clone()
        
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']
        _model = gtnet(True, True, 2,
                       device, predefined_A=None, 
                        dropout=0.3, subgraph_size=20,
                       node_dim=10, dilation_exponential=1,
                      seq_length=12, in_dim=in_dim, out_dim=12,
                      layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr_syn)
        min_val_loss = sys.float_info.max

        self.embedding.eval()
        with torch.no_grad():                    
            node_embed_syn = self.embedding(synx[..., 0])
            node_embed_syn = torch.mean(node_embed_syn, dim=0)  # num point x rank        
            
        for i in tqdm(range(200)):
            _model.train()
            _model.set_node_embed(node_embed_syn)
            output_syn = _model(synx.transpose(1,3)).squeeze()
            loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():
                    val_loss = _model.test_model(self.embedding, data['val_loader'], scaler, device)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            test_loss = _model.test_model(self.embedding, data['test_loader'], scaler, device)

        return min_i, min_val_loss, test_loss

    
    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny

        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']

        #min_i, val_loss, test_loss = self.test_syn()
        #print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        #with open(args.save_path, 'a') as f:
        #    f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        
        min_val_loss = sys.float_info.max
        optimizer = torch.optim.Adam([synx, syny] + list(self.embedding.parameters()), lr=args.lr_feat)
        for i in tqdm(range(args.epochs)):
            data['train_loader'].shuffle()
            data['train_loader'].current_ind = 0            
            _model = gtnet(True, True, 2, 
                  device, predefined_A=None, 
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
            model_params = list(_model.parameters())
            #_model.initialize()
            _model.to(self.device)
            _model.train()
            optimizer_model = torch.optim.Adam(model_params, lr=args.lr_syn)

            grad_loss = 0
            num_ol = 20
            num_real_total = 0
            for ol in range(num_ol):            
                self.embedding.train()
                # Compute real gradient                            
                x, y = data['train_loader'].get_next()                                
                realx = torch.tensor(x, device=self.device, dtype=torch.float)
                realy = torch.tensor(y, device=self.device, dtype=torch.float)
                node_embed = self.embedding(realx[..., 0])  #  batch size x num point x rank
                node_embed = torch.mean(node_embed, dim=0)  # num point x rank
                _model.set_node_embed(node_embed)
                
                realy = realy[:,:,:,0]  # batch x seq len x num node
                output_real_temp = _model(realx.transpose(1, 3)).squeeze()
                output_real = scaler.inverse_transform(output_real_temp)
                loss_real, num_real = util.masked_se(output_real, realy, 0.)
                gw_real = torch.autograd.grad(loss_real/num_real, model_params, retain_graph=True)
                gw_real = [_.detach().clone() for _ in gw_real]

                node_embed = self.embedding(synx[..., 0])
                node_embed = torch.mean(node_embed, dim=0)  # num point x rank
                _model.set_node_embed(node_embed)
                output_syn = _model(synx.transpose(1, 3)).squeeze()
                loss_syn = F.mse_loss(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)
                
                #pbar.close()
                _loss = util.match_loss(gw_syn, gw_real, self.device)                
                # gradient descent                
                grad_loss += _loss.item()
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                if ol == num_ol - 1:
                    break
                    
                num_il = 10
                self.embedding.eval()
                synx_in, syny_in = synx.detach(), syny.detach()
                with torch.no_grad():
                    node_embed = self.embedding(synx_in[..., 0])
                node_embed = torch.mean(node_embed, dim=0)  # num point x rank
                _model.set_node_embed(node_embed)
                for il in range(num_il):
                    optimizer_model.zero_grad()
                    output_syn_in = _model(synx_in.transpose(1,3)).squeeze()
                    loss_syn_in = F.mse_loss(output_syn_in, syny_in)
                    loss_syn_in.backward()
                    optimizer_model.step()
                    
            if (i+1) % 10 == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"my epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path + ".txt", 'a') as f:
                    f.write(f"my epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    synx_ = synx.detach().clone().cpu()
                    syny_ = syny.detach().clone().cpu()                    
                    torch.save({'x':synx_, 'y':syny_}, args.save_path + ".pt")
            else:
                print(f"epoch: {i}, grad loss: {grad_loss/num_ol}")


# python -m DC.distill_mtgnn2 -de 1 -e 300 -sp results/dc_mtgnn2_sr2e-2_nr1e-1 -lrf 1e-2 -lrs 0.01 -sr 2e-2 -nr 1e-1
# python -m DC.distill_mtgnn2 -de 3 -d ../data/PEMS-BAY -e 300 -sp results/dc_mtgnn2_pems_2e-3 -lrf 1e-2 -lrs 0.01 -r 2e-3
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**10, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate')
    parser.add_argument('-sr', '--series_reduce_rate',type=float,default=2e-2,help='learning rate')
    parser.add_argument('-nr', '--node_reduce_rate',type=float,default=1e-1,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-sp', '--save_path', type=str, default='results/') 
    parser.add_argument('-nh', '--nhid', type=int, default=32, help='')
    parser.add_argument('-dr', '--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('-sl', '--seq_length', type=int, default=12, help='')
    
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    #device = torch.device(f"cpu")
    dataloader =  util.load_dataset(args.data, 128)
    print("load finish")

    algo = GradMatch(dataloader, args, device)    
    algo.train()