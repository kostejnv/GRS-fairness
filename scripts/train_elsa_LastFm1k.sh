#!/usr/bin/env bash

python train_elsa.py --batch_size 64 --factors 1024 --epochs 100 --early_stop 10 --dataset LastFM1k --seed 42 --lr 0.0001