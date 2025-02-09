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


class GradMatch:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate *  data['train_loader'].xs.shape[0])
        scaler = data['scaler']
      
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

        self.node2cluster = np.load(args.mapping_path)
        self.num_clusters = len(set(self.node2cluster))
        num_node = self.node2cluster.shape[0]
        self.cluster2node = [[] for _ in range(self.num_clusters)]        
        for _n in range(num_node):
            self.cluster2node[self.node2cluster[_n]].append(_n)
        self.cluster2weight = [len(self.cluster2node[cl])/num_node for cl in range(self.num_clusters)]
        

    def train_gtnet(self):
        args = self.args
        num_nodes = self.data['train_loader'].xs.shape[2]
        in_dim = self.data['train_loader'].xs.shape[3]
        scaler = self.data['scaler']
        self.trained_model = gtnet(True, True, 1, num_nodes,
                  device, predefined_A=None,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)     
        self.trained_model.to(self.device)

        _optimizer = optim.Adam(self.trained_model.parameters(), lr=0.01)
        min_val_mse = sys.float_info.max
        for i in range(10):
    
            self.trained_model.train()
            self.data['train_loader'].shuffle()
            for iter, (x, y) in enumerate(self.data['train_loader'].get_iterator()):
                trainx = torch.tensor(x, device=device, dtype=torch.float)
                trainx = trainx.transpose(1, 3)
                trainy = torch.tensor(y, device=device, dtype=torch.float)
                trainy = trainy[:,:,:,0]
                output = self.trained_model(trainx).squeeze()
                output = scaler.inverse_transform(output)
                curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
                curr_loss /= num_val_entry
    
                _optimizer.zero_grad()
                curr_loss.backward()
                _optimizer.step()
    
            self.trained_model.eval()
            with torch.no_grad():               
                val_mse = self.trained_model.test_model(dataloader['val_loader'], scaler, device)
                test_mse = self.trained_model.test_model(dataloader['test_loader'], scaler, device)            
                if min_val_mse > val_mse:
                    min_val_mse = val_mse
                    weights = copy.deepcopy(self.trained_model.state_dict())
                print(f'epoch: {i}, valid mse: {val_mse}, test mse: {test_mse}')
                
        self.trained_model.load_state_dict(weights)

    
    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx.detach().clone()
        syny = self.syny.detach().clone()

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']
        static_feat = self.trained_model.gc.emb1.weight.clone().detach()
        _model = gtnet(True, True, 2, num_nodes,
                  device, predefined_A=None, static_feat=static_feat,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,
                  seq_length=12, in_dim=in_dim, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
        _model.to(self.device)
        optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(300)):
            _model.train()
            output_syn = _model(synx.transpose(1,3)).squeeze()
            loss_syn = F.mse_loss(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.eval()
            if (i+1)%10 == 0:
                with torch.no_grad():
                    val_loss = _model.test_model(data['val_loader'], scaler, device)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.state_dict())

        _model.load_state_dict(min_params)
        _model.eval()
        with torch.no_grad():
            test_loss = _model.test_model(data['test_loader'], scaler, device)

        return min_i, min_val_loss, test_loss

    
    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']

        #min_i, val_loss, test_loss = self.test_syn()
        #print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        #with open(args.save_path, 'a') as f:
        #    f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        
        min_val_loss = sys.float_info.max
        optimizer = torch.optim.Adam([synx, syny], lr=args.lr_feat)
        static_feat = self.trained_model.gc.emb1.weight.clone().detach()
        for i in tqdm(range(args.epochs)):
            data['train_loader'].shuffle()
            data['train_loader'].current_ind = 0            
            _model = gtnet(True, True, 2, num_nodes,
                  device, predefined_A=None, static_feat=static_feat,
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
                output_syn = _model(synx.transpose(1, 3)).squeeze()
                loss_syn = F.mse_loss(output_syn, syny)
                gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

                # Compute real gradient                            
                x, y = data['train_loader'].get_next()
                realx = torch.tensor(x, device=self.device, dtype=torch.float)
                realy = torch.tensor(y, device=self.device, dtype=torch.float)
                realy = realy[:,:,:,0]  # batch x seq len x num node
                output_real_temp = _model(realx.transpose(1, 3)).squeeze()
                output_real = scaler.inverse_transform(output_real_temp)

                _loss = 0
                for cl in range(self.num_clusters):
                    # synthetic grad
                    loss_syn = util.mse(output_syn[:, :, self.cluster2node[cl]], syny[:,:,self.cluster2node[cl]])
                    gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

                     # real grad
                    loss_real, num_real = util.masked_se(output_real[:, :, self.cluster2node[cl]], 
                                                         realy[:,:,self.cluster2node[cl]], 0.)


                    if cl < self.num_clusters - 1:
                        gw_real = torch.autograd.grad(loss_real/num_real, model_params, retain_graph=True)
                    else:
                        gw_real = torch.autograd.grad(loss_real/num_real, model_params)
                    gw_real = [_.detach().clone() for _ in gw_real]
                    _loss += self.cluster2weight[cl] * util.match_loss(gw_syn, gw_real, self.device)
                
                #pbar.close()
                grad_loss += _loss.item()
                            
                # gradient descent
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                if ol == num_ol - 1:
                    break
                    
                num_il = 5
                synx_in, syny_in = synx.detach(), syny.detach()
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
                    if _model.gc.static_feat is None:
                        node1 = _model.gc.emb1.weight.detach().clone().cpu()
                    else:
                        node1 = _model.gc.static_feat
                    torch.save({'x':synx, 'y':syny, 'node1': node1}, args.save_path + ".pt")
            else:
                print(f"epoch: {i}, grad loss: {grad_loss/num_ol}")

            
# python -m DC_cluster.distill_mtgnn2 -de 2 -e 300 -sp results/dc_mtgnn_clus2_1e-2.txt -lrf 0.01 -lrs 0.01 -r 1e-2
# python -m DC.distill_mtgnn -de 6 -d ../data/PEMS-BAY -e 1000 -sp results/dc_pems_mtgnn2 -lrf 0.001 -lrs 0.001 -r 3e-4
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-mp', '--mapping_path', type=str, default='DC_cluster/METR-LA.npy', help='data path')
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lrs', '--lr_syn',type=float,default=1e-2,help='learning rate')
    parser.add_argument('-lrf', '--lr_feat',type=float,default=0.1,help='learning rate')
    parser.add_argument('-r', '--reduction_rate',type=float,default=1e-3,help='learning rate')
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
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")

    algo = GradMatch(dataloader, args, device)
    algo.train_gtnet()
    algo.train()