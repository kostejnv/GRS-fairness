#!/bin/bash


sae_run_ids=(
'cf0be8cd67814a928d7229ffcdf162dd'
'2b5a599fc8d5402b94992cff58b93cbf'
'86d4e5f49f8444f39a29145427f716d8'
'f1736e15ef284778b2363c7cf8bbdbb5'
'f6e7e1c2a0b145869d8608d644e9cca9'
'316a7224bf7240b89258608172c562ca'
)

group_types=(
    'sim'
    'div'
    'random'
)

note="variants"

for sae_run_id in "${sae_run_ids[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python analyze_group_embedding_intersection.py --dataset MovieLens --sae_run_id "$sae_run_id" --group_type "$group_type" --group_size 5 --user_set test --note "$note"
    done
done