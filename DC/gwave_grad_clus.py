from gwave_grad import GwaveGrad
import numpy as np
import torch
import util
from tqdm import tqdm
from model import gwnet


class GwaveGradClus(GwaveGrad):

    def __init__(self, data, args, device):
        super().__init__(data, args, device)
        self.node2cluster = np.load(args.mapping_path)
        self.cluster2node = [[] for _ in range(args.num_clusters)]        
        num_node = self.node2cluster.shape[0]
        for _n in range(num_node):
            self.cluster2node[self.node2cluster[_n]].append(_n)
        self.cluster2weight = [len(self.cluster2node[cl])/num_node for cl in range(args.num_clusters)]
    

    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']

       # min_i, val_loss, test_loss = self.test_syn()
        #print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        #with open(args.save_path, 'a') as f:
        #    f.write(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")        
        optimizer = torch.optim.Adam([synx, syny], lr=args.learning_rate)
        for i in tqdm(range(args.epochs)):
            if i == args.epochs//2:
                optimizer = torch.optim.Adam([synx, syny], lr=args.learning_rate/10)
                         
            _model = gwnet(self.device, num_nodes, 0, in_dim, args.seq_length, 
                               residual_channels=args.nhid, dilation_channels=args.nhid, 
                               skip_channels=8*args.nhid, end_channels=16*args.nhid)
            model_params = list(_model.parameters())
            _model.initialize()
            _model.to(self.device)
            _model.train()
            optimizer_model = torch.optim.Adam(model_params, lr=0.0001)

            train_loss = 0
            num_ol = 20
            num_real_total = 0
            for ol in range(num_ol):  
                data['train_loader'].shuffle()
                data['train_loader'].current_ind = 0   
                # compute synthetic gradient
                output_syn = _model(synx.transpose(1, 3)).squeeze()  # batch x seq len x num point

                # Compute real gradient                            
                x, y = data['train_loader'].get_next()
                realx = torch.tensor(x, device=self.device, dtype=torch.float)
                realx = realx.transpose(1, 3)
                realy = torch.tensor(y, device=self.device, dtype=torch.float)
                realy = realy[:,:,:,0]
                output_real = _model(realx).squeeze()
                output_real = scaler.inverse_transform(output_real)

                # Gradient matching
                _loss = 0
                for cl in range(args.num_clusters):                    
                    # synthetic grad
                    loss_syn = util.mse(output_syn[:, :, self.cluster2node[cl]], syny[:,:,self.cluster2node[cl]])
                    gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

                    # real grad
                    loss_real, num_real = util.masked_se(output_real[:, :, self.cluster2node[cl]], 
                                                         realy[:,:,self.cluster2node[cl]], 0.)

                                            
                    gw_real = torch.autograd.grad(loss_real/num_real, model_params, retain_graph=True)
                    #    gw_real = torch.autograd.grad(loss_real/num_real, model_params, retain_graph=True)
                    gw_real = list((_.detach().clone() for _ in gw_real))      
                    _loss += self.cluster2weight[cl] * util.match_loss(gw_syn, gw_real, self.device)
                #pbar.close()                
                
                # gradient descent
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                if ol == num_ol - 1:
                    break
                    
                num_il = 5
                for il in range(num_il):
                    x, y = data['train_loader'].get_next()
                    realx = torch.tensor(x, device=self.device, dtype=torch.float)
                    realx = realx.transpose(1, 3)
                    realy = torch.tensor(y, device=self.device, dtype=torch.float)
                    realy = realy[:,:,:,0]
                    output_real = _model(realx).squeeze()
                    output_real = scaler.inverse_transform(output_real)

                    # Compute real gradient                                                
                    loss_real_in, num_real_in = util.masked_se(output_real, realy, 0.)
                    loss_real_in /= num_real_in
                    optimizer_model.zero_grad()
                    loss_real_in.backward()
                    optimizer_model.step()
                    
            if (i+1) % 10 == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}\n")
            else:
                print(f"epoch: {i}, train loss: {_loss}")