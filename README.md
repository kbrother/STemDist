# Effective Dataset Distillation for Spatio-Temporal Forecasting with Bi-dimensional Compression

This repository is the official implementation of Effective Dataset Distillation for Spatio-Temporal Forecasting with Bi-dimensional Compression.

## Requirements
Please see the requirements.txt
```
fast_pytorch_kmeans==0.2.2
matplotlib==3.10.7
numpy==2.3.4
scikit_learn==1.7.2
scipy==1.16.3
torch==2.2.0
tqdm==4.65.0
```

## Input formats
Please download and check the datasets below for more details.

* There are three npz files (`train.npz, val.npz, test.npz`) per dataset.
* Each file contains two arrays, `x` and `y`. `x` is an array of input time series, and `y` is an array of target time series.

## Running STemDist
The distillation process of STemDist is implemented in `stemdist.py`.
### Positional arguments
* `-de`, `--device`: GPU id for execution.
* `-d`, `--data`: Location of the dataset folder.
* `-b`, `--batch_size`: Batch size for the distillation process.
* `-lrs`, `--lr_syn`: Learning rate for the surrogate model, which is trained on the synthetic dataset.
* `-lrf`, `--lr_feat`: Learning rate for the synthetic dataset.
* `-nrr`, `--node_reduce_rate`: Compression ratio for the spatial dimension.
* `-srr`, `--sereis_reduce_rate`: Compression ratio for the temporal dimension.
* `-e`, `--epoch`: Number of outer iterations.
* `-ned`, `--ne_dim`: Hidden dimension of the location embedding model.
* `-s`, `--seed`: Seed of execution.
* `-sp`, `--save_path`: Path for saving the result files.
* `-c`, `--check_freq`: Period in outer iterations for checking the performance of the distilled dataset.

### Example command
```
  python -m stemdist -de 0 -d ../data/GBA -e 100 -sp results/dc_dsa_cluster_gba_1e-3_1e-3 -lrf 1e-3 -lrs 1e-3 -srr 0.1 -nrr 0.1 -b 256 -ned 32 -s 0 -c 5
```


