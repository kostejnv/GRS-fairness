#!/usr/bin/env bash

# Run the Python script three times with different top_k values
python train_sae.py --embedding_dim 2048 --dataset EchoNest --top_k 16 --base_run_id d3653aa37bea49ec9917c479300aa2f3
python train_sae.py --embedding_dim 2048 --dataset EchoNest --top_k 64 --base_run_id d3653aa37bea49ec9917c479300aa2f3
python train_sae.py --embedding_dim 2048 --dataset EchoNest --top_k 256 --base_run_id d3653aa37bea49ec9917c479300aa2f3

# python train_sae.py --embedding_dim 8192 --dataset EchoNest --top_k 16 --base_run_id d3653aa37bea49ec9917c479300aa2f3
# python train_sae.py --embedding_dim 8192 --dataset EchoNest --top_k 64 --base_run_id d3653aa37bea49ec9917c479300aa2f3
# python train_sae.py --embedding_dim 8192 --dataset EchoNest --top_k 256 --base_run_id d3653aa37bea49ec9917c479300aa2f3