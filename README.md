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
