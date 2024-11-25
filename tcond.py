import torch.nn as nn
from tqdm import tqdm
from model import gwnet
import torch
import util
import sys
import copy


class TCond:

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.num_elems = int(args.reduction_rate *  data['train_loader'].xs.shape[0])

        # Define condensed data
        _shape = list(data['train_loader'].xs.shape[1:])
        _shape = tuple([self.num_elems] + _shape)
        self.synx = torch.rand(_shape, device=device, dtype=torch.float)
        self.synx = nn.Parameter(self.synx)

        _shape = list(data['train_loader'].ys.shape[1:-1])
        _shape = tuple([self.num_elems] + _shape)
        self.syny = torch.rand(_shape, device=device, dtype=torch.float)
        self.syny = nn.Parameter(self.syny)
        print(f'feat x shape: {self.synx.shape}')
        print(f'feat y shape: {self.syny.shape}')


    def test_syn(self):
        args = self.args
        data = self.data
        synx = self.synx.detach().clone()
        syny = self.syny.detach().clone()

        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']
        _model = gwnet(self.device, num_nodes, args.dropout, in_dim, args.seq_length, 
                           residual_channels=args.nhid, dilation_channels=args.nhid, 
                           skip_channels=8*args.nhid, end_channels=16*args.nhid)
        _model.to(self.device)
        _model = nn.DataParallel(_model, device_ids=[0,1,2,3])
        optimizer = torch.optim.Adam(_model.module.parameters(), lr=0.001, weight_decay=0.0001)
        min_val_loss = sys.float_info.max
        for i in tqdm(range(200)):
            _model.module.train()
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            loss_syn = util.mae(output_syn, syny)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

            _model.module.eval()
            if (i+1)%20 == 0:
                with torch.no_grad():
                    val_loss = _model.module.test_model(data['val_loader'], scaler)
    
                if min_val_loss > val_loss:
                    min_i = i
                    min_val_loss = val_loss
                    min_params = copy.deepcopy(_model.module.state_dict())

        _model.module.load_state_dict(min_params)
        _model.module.eval()
        with torch.no_grad():
            test_loss = _model.module.test_model(data['test_loader'], scaler)

        return min_i, min_val_loss, test_loss
                
    
    def train(self):
        args = self.args
        data = self.data
        synx, syny = self.synx, self.syny

        print(data['train_loader'].xs.shape[0])
        num_nodes = data['train_loader'].xs.shape[2]
        in_dim = data['train_loader'].xs.shape[3]
        scaler = data['scaler']


        min_i, val_loss, test_loss = self.test_syn()
        print(f"initial, min i: {min_i}, val loss: {val_loss}, test loss: {test_loss}")
        
        optimizer = torch.optim.Adam([synx, syny], lr=args.learning_rate)
        for i in tqdm(range(args.epochs)):
            data['train_loader'].shuffle()
            _model = gwnet(self.device, num_nodes, 0, in_dim, args.seq_length, 
                               residual_channels=args.nhid, dilation_channels=args.nhid, 
                               skip_channels=8*args.nhid, end_channels=16*args.nhid)
            model_params = list(_model.parameters())
            _model.to(self.device)
            _model = nn.DataParallel(_model, device_ids=[0,1,2,3])
            _model.module.train()

            #pbar = tqdm(total=data['train_loader'].xs.shape[0])
            # compute synthetic gradient
            output_syn = _model(synx.transpose(1, 3)).squeeze()
            loss_syn = util.mae(output_syn, syny)
            gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

            num_real = 0
            gw_real = []
            for x, y in data['train_loader'].get_iterator():                
                # Compute real gradient                            
                realx = torch.tensor(x, device=self.device, dtype=torch.float)
                realx = realx.transpose(1, 3)
                realy = torch.tensor(y, device=self.device, dtype=torch.float)
                realy = realy[:,:,:,0]
                output_real = _model(realx).squeeze()
                output_real = scaler.inverse_transform(output_real)
                loss_real, curr_num_real = util.masked_ae(output_real, realy, 0.)
                gw_real_curr = torch.autograd.grad(loss_real, model_params)
                gw_real_curr = list((_.detach().clone() for _ in gw_real_curr))
                num_real += curr_num_real
                
                for j, gw in enumerate(gw_real_curr):
                    if len(gw_real) < len(gw_real_curr):
                        gw_real.append(gw)
                    else:
                        gw_real[j] = gw_real[j] + gw
                #pbar.update(x.shape[0])

            for j in range(len(gw_real)):
                gw_real[j] /= num_real
                
            #pbar.close()
            _loss = util.match_loss(gw_syn, gw_real, self.device)
            train_loss = _loss.item()
            # gradient descent
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:                
                min_i, val_loss, test_loss = self.test_syn()
                print(f"epoch: {i}, min i: {min_i}, train_loss: {train_loss}, val loss: {val_loss}, test loss: {test_loss}")
                with open(args.save_path, 'a') as f:
                    f.write(f"epoch: {i}, min i: {min_i}, train_loss: {train_loss}, val loss: {val_loss}, test loss: {test_loss}\n")
            else:
                print(f"epoch: {i}, train loss: {train_loss}")
                