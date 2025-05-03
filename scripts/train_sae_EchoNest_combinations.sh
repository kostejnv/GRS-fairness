#!/usr/bin/env bash

# Best combinations

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 16 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 16 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 16 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 16 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0 --sample_users

python train_sae.py --dataset EchoNest --base_run_id 494195a6c97f49169010f64a3bfcdf2a --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 16 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0 --normalize

# --------------------------------
# New experiments

