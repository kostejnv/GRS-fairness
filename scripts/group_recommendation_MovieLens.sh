#!/bin/bash

# variable note
note="sizes_without_acts_v2"

sae_run_ids=(
'738b64ba61a64752b4edba280a4eef9e'
)


group_types=(
    'outlier'
)

user_types=(
    'test'
    # 'train'
)

fusion_strategies=(
    # 'at_least_2_common_features'
    # 'average'
    # 'square_average'
    # 'topk'
    # 'max'
    # 'common_features'
    'wcom'
)

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        for sae_run_id in "${sae_run_ids[@]}"
        do
            for fusion_strategy in "${fusion_strategies[@]}"
            do
                python recommend_for_groups.py --dataset MovieLens --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --note "$note" --group_set test --topk_inference
            done
        done
    done
done