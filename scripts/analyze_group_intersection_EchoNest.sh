#!/bin/bash


sae_run_ids=(
'a66b707390e44043a4352b039b55b0c5'
'9db2181c9b854dc7b010c389e63ff1a1'
'b95eb0a7205d491f927fde9058a454fe'
'cf31d085eb7f418a8f93fa0a9b50749b'
'8e34d4187a6844edb1d3c0cb151c09f8'
'24e747c2878544b0af4c19dc05aa26e5'
'38792c32ca094e259a04fd91eb954a4a'
'b03609b7b5eb472ab309c001724b8a39'
'836e700644eb457abd2b4b265d198736'
)

group_types=(
    'sim'
    'div'
    'random'
)

for sae_run_id in "${sae_run_ids[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python analyze_group_embedding_intersection.py --dataset EchoNest --sae_run_id "$sae_run_id" --group_type "$group_type" --group_size 3 --user_set test
    done
done