#!/bin/bash

sae_run_ids=(
    'f94c2b71ea6f49ea8b7f15ec7bcd898f'
    '23066f676669401ba3a51ec80ee67611'
    '187b834845454facbc47846538884165'
    '1401a31beaba40a4b4605dd0c8cfdd16'
    '253f2ec70dec4c05be3696b2356b80fb'
    'fcc282c69ba8421189f1ff7a6c23029b'
    'd07c9ad6b69540778a1cd53099b91670'
    '312d5703119b4d5abb359579774c26f4'
    'db01455e879a446bbc53c6ef3e0ca723'
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
        python analyze_group_embedding_intersection.py --dataset LastFM1k --sae_run_id "$sae_run_id" --group_type "$group_type" --group_size 3 --user_set train
    done
done