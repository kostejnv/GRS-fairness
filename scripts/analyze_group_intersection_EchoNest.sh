#!/bin/bash

sae_run_ids=(
    '73669f586bc8439bab330a71ae707577'
    '58539403c988473aa6a4761302d5b7de'
    'c488a3a255bf4344ae182274da2176af'
    'c829c0df33b04158aec242c7daa94c38'
    '4f4793e0c5fa4045a2f01c8c6a59a49d'
    '7dcdc7663adf47f5bb9b06aca5fca746'
    'e08194a6a2434b5785f688ac82827aed'
    '2b5970af37bf4172afdfad5c8df11101'
    '77fc5ef2506d4f33aba10554f47d6a46'
)

group_types=(
    'sim'
    'div'
    '21'
)

user_types=(
    'test'
    'train'
)

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        for sae_run_id in "${sae_run_ids[@]}"
        do
            python analyze_group_embedding_intersection.py --dataset EchoNest --sae_run_id "$sae_run_id" --group_type "$group_type" --group_size 3 --user_set "$user_type"
        done
    done
done