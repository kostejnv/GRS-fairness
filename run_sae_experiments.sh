#!/usr/bin/env bash

# Run the Python script three times with different top_k values
python train_sae.py --embedding_dim 8096 --dataset EchoNest --top_k 16
python train_sae.py --embedding_dim 8096 --dataset EchoNest --top_k 64
python train_sae.py --embedding_dim 8096 --dataset EchoNest --top_k 256