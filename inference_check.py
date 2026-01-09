import argparse
import torch
from model.mtgnn import gtnet
import time


# python -m inference_check -de 0 -nl 2352 -nf 1 -sl 12 -sp results/gba.txt
# python -m inference_check -de 0 -nl 3834 -nf 1 -sl 12 -sp results/gla.txt
# python -m inference_check -de 0 -nl 6561 -nf 6 -sl 12 -sp results/era5.txt
# python -m inference_check -de 0 -nl 7070 -nf 6 -sl 12 -sp results/cams.txt
# python -m inference_check -de 0 -nl 8600 -nf 1 -sl 12 -sp results/ca.txt
# python -m inference_check -de 0 -nl 32768 -ni 8 -nf 1 -sl 12 -sp results/max_loc.txt 
# python -m inference_check -de 0 -nl 2352 -nf 8192 -sl 12 -sp results/max_feat.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-nl', '--num_loc', type=int, default=2352, help='number of location')
    parser.add_argument('-nf', '--num_feat', type=int, default=6, help='number of location')
    parser.add_argument('-sl', '--seq_len', type=int, default=12, help='number of location')
    parser.add_argument('-ni', '--num_iter', type=int, default=1, help='number of location')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='number of location')
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}")

    _model = gtnet(True, True, 2, args.num_loc, 
                  device, predefined_A=None, use_static_feat=True,
                  dropout=0.3, subgraph_size=20,
                  node_dim=10, dilation_exponential=1,             
                  seq_length=args.seq_len, in_dim=args.num_feat, out_dim=1,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, ne_dim=32)

    _model.to(device)

    input_size = args.num_feat * args.seq_len

    with torch.no_grad():
        for i in range(10):
    
            time_check = 0
            for j in range(args.num_iter):
                nm_input = torch.rand(args.num_loc, input_size).to(device)
                _input = torch.rand(args.batch_size, args.seq_len, args.num_loc, args.num_feat).to(device)
                _input = _input.transpose(1, 3)
    
                start_time = time.time()
                _model.embed_forward(nm_input)
                _output = _model(_input)
                time_check += time.time() - start_time
                print(j)
    
            if i >= 5:
                print(time_check)
                with open(args.save_path, "a") as f:
                    f.write(f"{time_check}\n")

    