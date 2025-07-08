#!/usr/bin/env bash

python train_elsa.py --batch_size 64 --factors 256 --epochs 100 --early_stop 10 --dataset LastFM1k --seed 42 --lr 0.0001
python train_elsa.py --batch_size 1024 --factors 256 --epochs 100 --early_stop 10 --dataset MovieLens --seed 42 --lr 0.0001