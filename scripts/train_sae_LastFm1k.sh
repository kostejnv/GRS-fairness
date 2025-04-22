#!/usr/bin/env bash

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 256 --reconstruction_loss Cosine --auxiliary_coef 0 --contrastive_coef 0

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 256 --reconstruction_loss L2 --auxiliary_coef 0 --contrastive_coef 0

python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 256 --reconstruction_loss Cosine --auxiliary_coef 0 --contrastive_coef 0 --normalize

python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 256 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 256 --reconstruction_loss Cosine --auxiliary_coef 0 --contrastive_coef 0.3

# embedding_dim 2048

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 16

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 64

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 256

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 512

# # test withou sample users

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --epochs 10000 --early_stop 100 --seed 42 --embedding_dim 2048  --top_k 256

# # embedding_dim 8192

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 100 --seed 42 --embedding_dim 8192  --top_k 16

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 100 --seed 42 --embedding_dim 8192  --top_k 64

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 100 --seed 42 --embedding_dim 8192  --top_k 256

# python train_sae.py --dataset LastFM1k --base_run_id 34ade4833e9e48d9b2d3c504a0af4346 --batch_size 64 --sample_users --epochs 10000 --early_stop 100 --seed 42 --embedding_dim 8192  --top_k 512