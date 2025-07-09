#!/usr/bin/env bash

python train_elsa.py --batch_size 1024 --factors 1024 --epochs 100 --early_stop 10 --dataset EchoNest --seed 42 --lr 0.0001