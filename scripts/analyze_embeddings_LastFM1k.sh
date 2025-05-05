#!/bin/bash

sae_run_ids=(
    'db01455e879a446bbc53c6ef3e0ca723'
)

for sae_run_id in "${sae_run_ids[@]}"
do
    python analyze_embedding.py --dataset LastFM1k --sae_run_id "$sae_run_id" --user_set train --user_sample 0
done