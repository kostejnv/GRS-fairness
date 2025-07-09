#!/bin/bash
experiment_id=657713966175362303 # Change to your run
note=sizes_L2
note=all_sagea


group_types=(
    'sim'
    'outlier'
    'random'
)

fusion_strategies=(
    'average'
    'topk'
    'max'
    'common_features'
    'wcom'
)

for group_type in "${group_types[@]}"
do
    for strategy in "${fusion_strategies[@]}"
    do
        python recommend_all_runs.py --note "$note_item" --experiment_id "$experiment" --group_type "$group_type" --fusion_strategy "$strategy" --topk_inference --true_note "$true_note" # variants when the TopK acitvation is on during inference
        python recommend_all_runs.py --note_to_filter "$note_item" --experiment_id "$experiment" --group_type "$group_type" --fusion_strategy "$strategy" --note  "$note" # variants when the TopK acitvation is on during inference
    done
done