#!/usr/bin/env bash

# embedding_dim 2048

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --sample_users --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 16

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --sample_users --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 64

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --sample_users --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 256

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --sample_users --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 512

# test withou sample users

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 256

# embedding_dim 8192

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --sample_users --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 8192  --top_k 16

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --sample_users --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 8192  --top_k 64

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --sample_users --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 8192  --top_k 256

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --sample_users --epochs 1000 --early_stop 100 --seed 42 --embedding_dim 8192  --top_k 512