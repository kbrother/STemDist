import argparse
from utils import *
from model import AGCRN
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


def test_model(model, _dataloader, scaler, device):
    model.module.eval()
    total_loss = 0
    num_entry = 0
    with torch.no_grad():
        for b_data, b_target in _dataloader:
            b_data = torch.stack(tuple(b_data), dim=0).to(device)
            label = torch.stack(tuple(b_target), dim=0).to(device)
            label = label[..., 0]
            output = model(b_data).squeeze()

            output = scaler.inverse_transform(output)
            label = scaler.inverse_transform(label)
            curr_loss = F.l1_loss(output, label, reduction='sum')
            total_loss += curr_loss.item()
            num_entry += torch.numel(label)
    
    return total_loss / num_entry        
    

# python train_agcrn.py -de 0 
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('-de', '--device', type=int, default=0, help='')
    args.add_argument('-d', '--dataset', type=str, default='PEMS04', help='data path')
    args.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    args.add_argument('-r', '--rnn_units', type=int, default=2**6, help='rnn hidden unit')
    args.add_argument('-nl', '--num_layers', default=2, type=int)
    args.add_argument('-ed', '--embed_dim', default=10, type=int)
    args.add_argument('-lr', '--lr', default=0.003, type=float)
    args.add_argument('-e', '--epochs', default=100, type=int)
    args = args.parse_args()    

    train_loader, val_loader, test_loader, scaler, _data = get_dataloader(args)

    num_nodes, input_dim = _data.shape[1], _data.shape[2]
    model = AGCRN(args, num_nodes, input_dim)
    #init loss function, optimizer    

    device = torch.device(f"cuda:{args.device}")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    model = model.to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    for e in tqdm(range(args.epochs)):
        model.module.train()
        train_loss, num_entry = 0, 0
        for b_data, b_target in train_loader:
            b_data = torch.stack(tuple(b_data), dim=0).to(device)
            label = torch.stack(tuple(b_target), dim=0).to(device)
            label = label[..., 0]            
            optimizer.zero_grad()
            
            output = model(b_data).squeeze()
            curr_loss = F.l1_loss(output, label, reduction='sum')
            train_loss += curr_loss.item()
            num_entry += torch.numel(label)
            
            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()
        train_loss /= num_entry

        val_loss = test_model(model, val_loader, scaler, device)
        test_loss = test_model(model, test_loader, scaler, device)

        print(f'epoch: {e}, train loss: {train_loss}, val loss: {val_loss}, test loss: {test_loss}')

            