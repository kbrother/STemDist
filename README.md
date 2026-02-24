# Effective Dataset Distillation for Spatio-Temporal Forecasting with Bi-dimensional Compression

This repository is the official implementation of Effective Dataset Distillation for Spatio-Temporal Forecasting with Bi-dimensional Compression, Taehyung Kwon*, Yeonje Choi*, Yeongho Kim, and Kijung Shin, ICDE 2026 (to appear).

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
* `-srr`, `--series_reduce_rate`: Compression ratio for the temporal dimension.
* `-e`, `--epoch`: Number of outer iterations.
* `-ned`, `--ne_dim`: Hidden dimension of the location embedding model.
* `-s`, `--seed`: Seed of execution.
* `-sp`, `--save_path`: Path for saving the result files.
* `-c`, `--check_freq`: Period in outer iterations for checking the performance of the distilled dataset.

### Example command
```
  python -m stemdist -de 0 -d ../data/GBA -e 100 -sp results/stemdist_gba_1e-3_1e-3 -lrf 1e-3 -lrs 1e-3 -srr 0.1 -nrr 0.1 -b 256 -ned 32 -s 0 -c 5
```

### Example output
* `stemdist_gba_1e-3_1e-3.pt`: Saves the distilled dataset.
* `stemdist_gba_1e-3_1e-3.txt`: Saves the performance of distilled datasets for every 'check_freq' outer iteration.


## Checking the performance of the distilled dataset
Checking the performance of the distilled dataset is implemented in `model/load_mtgnn_stemdist.py`.

### Positional arguments
* `-de, -d, -s, -ned, -sp`: Same with the cases of running `stemdist.py`.
* `-b`, `--batch_size`: Batch size for the validation of the trained model.
* `-lr`, `--lr`: Learning rate for training the model.
* `-e`, `--epochs`: Number of training epochs for the model.
* `-c`, `--check`: Period in epochs for checking the performance of the trained model.
* `-lp`, `--load_path`: Path which saves the distilled dataset.
* `-a`, `--ae`: Compute error in Relative MSE when given.

### Example command
```
   python -m model.load_mtgnn_stemdist -de 0 -d ../data/GBA -lr 0.01 -e 400 -b 128 -lp results/stemdist_gba_1e-3_1e-3.pt -s 0 
```

## Real-world datasets which we used
|Name|M (# time series)|N (# locations)|F (# features)|# Total data points|Source|Downlaod Link|
|-|-|-|-|-|-|-|
|GBA|1,997|2,352|1|4,649,904|[PatchSTG](https://github.com/lmissher/patchstg)|[Link](https://www.dropbox.com/scl/fo/2y7wtmmgobsgj359kso3j/AKwXqwmRWifW9e_uArtMHEY?rlkey=8vkvhaj2y14mhif79hit2ae0k&st=hxb9yf8c&dl=0)|
|GLA|1,997|3,834|1|7,579,818|[PatchSTG](https://github.com/lmissher/patchstg)|[Link](https://www.dropbox.com/scl/fo/w756fegl6c0ek2xnvd61d/AGcHKDOTUeZriAnWgn8TJeU?rlkey=0klfypwm8xxqtpmxkzqpiuez7&st=ren9qisg&dl=0)|
|ERA5|2,137|6,561|6|14,020,857|[Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)|[Link](https://www.dropbox.com/scl/fo/w08d40g6z8l7nbqs4jie6/AATX1x8gc5b8NMp6V90arMU?rlkey=by3580rtr1bgt5dyb8cjsuj70&st=0rbxqueo&dl=0)|
|CAMS|2,556|7,070|6|108,425,520|[ECMWF](https://www.ecmwf.int/en/forecasts/dataset/cams-global-reanalysis)|[Link](https://www.dropbox.com/scl/fo/ul1hrzsbdabvt3exgbygh/AB9QZ0-BI-FvJF5JBxTdkXc?rlkey=wq9y9qnw7k4hor6gvdt8j4wx9&st=9dvuv795&dl=0)|
|CA|1,997|8,600|1|17,002,200|[PatchSTG](https://github.com/lmissher/patchstg)|[Link](https://www.dropbox.com/scl/fo/6kprmjr8p9v1yy1kg65wa/ALidoPgwv7AbmUp6J-sNYJw?rlkey=2k7niewn6c691u7b1e1mh0dxe&st=hkze3x4h&dl=0)|
