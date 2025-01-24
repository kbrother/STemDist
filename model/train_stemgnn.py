import torch
import numpy as np
import argparse
import time
import util
from tqdm import tqdm
import random
from model.stemgnn import StemGNN
import torch.optim as optim
import math


# python -m model.train_stemgnn -de 0 -d ../data/METR-LA -lr 1e-2 -e 100
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA-Tensor', help='data path')
    parser.add_argument('--multi_layer', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=2**8, help='batch size')
    parser.add_argument('-lr', '--learning_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    args = parser.parse_args()
    
    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader = util.load_dataset(args.data, args.batch_size)
    print("load finish")
    
    scaler = dataloader['scaler']
    num_nodes = dataloader['train_loader'].xs.shape[2]
    in_dim = dataloader['train_loader'].xs.shape[3]

    model = StemGNN(num_nodes, 2, 12, args.multi_layer, horizon=12)    
    model.to(device)
    #model.reset_parameters()
    
    _optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for i in range(args.epochs):

        model.train()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.tensor(x, device=device, dtype=torch.float)
            trainy = torch.tensor(y, device=device, dtype=torch.float)
            trainy = trainy[:,:,:,0]
            trainx = trainx[:,:,:,0]
            output, _ = model(trainx)            

            output = scaler.inverse_transform(output)
            curr_loss, num_val_entry = util.masked_se(output, trainy, 0.)
            curr_loss /= num_val_entry

            _optimizer.zero_grad()
            curr_loss.backward()
            _optimizer.step()

        model.eval()
        with torch.no_grad():               
            val_mse = model.test_model(dataloader['val_loader'], scaler, device)
            test_mse = model.test_model(dataloader['test_loader'], scaler, device)            
            print(f'epoch: {i}, valid mae: {val_mse}, test mae: {test_mse}')

