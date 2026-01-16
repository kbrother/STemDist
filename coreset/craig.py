import argparse
import torch
import random
import numpy as np
import util
from model.mtgnn import gtnet
from tqdm import tqdm


def get_grad(_model, _data, device):
    
    last_params = [p for p in _model.end_conv_2.parameters() if p.requires_grad]
    xs_orig = _data['train_loader'].xs_orig
    ys_orig = _data['train_loader'].ys_orig
    num_series = xs_orig.shape[0]
    scaler = _data['scaler']

    grads_flat = []
    for i in range(num_series):
        _model.zero_grad(set_to_none=True)
        curr_x = xs_orig[i].unsqueeze(0)
        curr_x = torch.tensor(curr_x, device=device)
        curr_x = curr_x.transpose(1, 3)
        curr_y = torch.tensor(curr_y, device=device)
        output = _model(curr_x).squeeze()
        output = scaler.inverse_transform(output)
        curr_loss, num_val_entry = util.masked_se(output, curr_y, 0.)
        curr_loss /= num_val_entry

        curr_loss.backward()
        gi = []
        for p in last_params:
            g = p.grad
            assert(g is not None)
            gi.append(g.detach().reshape(-1))
        grads_flat.append(torch.cat(gi, dim=0))

    return torch.stack(grads_flat, dim=0)
    

if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader =  util.load_dataset(args.data)

    _model = gtnet(True, True, 2, num_nodes, 
                      device, predefined_A=None, use_static_feat=False,
                      dropout=0.3, subgraph_size=20,
                      node_dim=10, dilation_exponential=1,             
                      seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                      layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
    _model = _model.to(device)
    get_grad(_model, dataloader, device)