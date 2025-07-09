#!/usr/bin/env bash

base_run_id=0c2c7c4b7cd5427db21b9c7022ffbc18 # Add your base run ID here
dataset=MovieLens

# basic different sizes

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0


python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0


python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0

# --------------------------------------------------
# Cosine reconstruction loss

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 32 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 64 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 128 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0


python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 32 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 64 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 128 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0


python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 32 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 64 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 128 --reconstruction_loss Cosine --auxiliary_coef 0.03125 --contrastive_coef 0

# --------------------------------------------------
# Contrastive 0.3

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3


python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3


python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.3

# --------------------------------------------------
# Contrastive 0.5

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 1024  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5


python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 2048  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5


python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 32 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 64 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5

python train_sae.py --dataset "$dataset" --base_run_id "$base_run_id" --batch_size 1024 --epochs 4000 --early_stop 250 --seed 42 --embedding_dim 4096  --top_k 128 --reconstruction_loss L2 --auxiliary_coef 0.03125 --contrastive_coef 0.5