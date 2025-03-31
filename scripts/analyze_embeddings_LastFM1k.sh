#!/bin/bash

sae_run_ids=(
    'c0e0c953b7394bbaabaac917a9274beb'
    '5b045a645c794a51b0d28c317e3c1e4d'
    '4ffd4c831f0343f3b96e1bf7ff27fb05'
    'b1d64424a40740df844e914e1ed12568'
    '75df48f60ec24ba68265c003781c3deb'
    'c4e39f27c0964f5cae0f52f76cd46dd6'
    '548ebfc2219148f28695eda707ca4566'
    'b3ab3fa715d84660a35ce428348216dc'
    'c1f21aa623cb4b838d296536b2cb5989'
)

for sae_run_id in "${sae_run_ids[@]}"
do
    python analyze_embedding.py --dataset LastFM1k --sae_run_id "$sae_run_id" --user_set full --user_sample 0
done