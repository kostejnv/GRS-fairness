#!/usr/bin/env bash

dataset=MovieLens


if [ "$dataset" = "MovieLens" ]; then
    batch_size=1024
elif [ "$dataset" = "LastFM1k" ]; then
    batch_size=64
else
    echo "Unknown dataset: $dataset"
    exit 1
fi

python train_elsa.py --batch_size $batch_size --factors 256 --epochs 100 --early_stop 10 --dataset $dataset --seed 42 --lr 0.0001