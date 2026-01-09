import argparse
from model.node_embed import NodeEmbedding_attn
import time
import torch


# python -m latency_check -de 0 -nl 2352 -nf 1 -sl 12 -sp results/gba.txt
# python -m latency_check -de 0 -nl 3834 -nf 1 -sl 12 -sp results/gla.txt
# python -m latency_check -de 0 -nl 6561 -nf 6 -sl 12 -sp results/era5.txt
# python -m latency_check -de 0 -nl 7070 -nf 6 -sl 12 -sp results/cams.txt
# python -m latency_check -de 0 -nl 8600 -nf 1 -sl 12 -sp results/ca.txt
# python -m latency_check -de 0 -nl 32768 -ni 8 -nf 1 -sl 12 -sp results/max_loc.txt 
# python -m latency_check -de 0 -nl 2352 -nf 8192 -sl 12 -sp results/max_feat.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-nl', '--num_loc', type=int, default=2352, help='number of location')
    parser.add_argument('-nf', '--num_feat', type=int, default=6, help='number of location')
    parser.add_argument('-sl', '--seq_len', type=int, default=12, help='number of location')
    parser.add_argument('-ni', '--num_iter', type=int, default=1, help='number of location')
    parser.add_argument('-sp', '--save_path', type=str, default='results/')
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}")

    input_size = args.num_feat * args.seq_len
    _model = NodeEmbedding_attn(input_size, 32, 10).to(device)
    _model.eval()
    with torch.no_grad():
        for i in range(10):             
            time_check = 0
            for j in range(args.num_iter):
                _input = torch.rand(args.num_loc, input_size).to(device)
                start_time = time.time()
                _model(_input)
                time_check += time.time() - start_time

            if i >= 5:
                print(time_check)
                with open(args.save_path, "a") as f:
                    f.write(f"{time_check}\n")
                    