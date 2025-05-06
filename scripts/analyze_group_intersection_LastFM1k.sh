#!/bin/bash

sae_run_ids=(
'91986527055744a798f3f785694ef9ac'
'e6f874617a574800b8d5a11254711223'
'4f7b3e314fc24e3b8f42b6ad72148888'
'539e93e0a9a247e291106aeaface5a32'
'6f9e049848024ae3a7a65af9bcd1ce7b'
'a8bd0fb5815f44e2a2e3205d140790f6'
)

group_types=(
    'sim'
    'div'
    'random'
)

note=variants

for sae_run_id in "${sae_run_ids[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python analyze_group_embedding_intersection.py --dataset LastFM1k --sae_run_id "$sae_run_id" --group_type "$group_type" --group_size 3 --user_set train --note "$note"
    done
done