#!/usr/bin/env bash

base_run_id=0c2c7c4b7cd5427db21b9c7022ffbc18

python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 16 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0


python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0


python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset MovieLens --base_run_id "$base_run_id" --batch_size 1024 --epochs 10000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0