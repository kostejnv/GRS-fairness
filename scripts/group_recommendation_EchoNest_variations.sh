#!/bin/bash

# variable note
note="variations"

sae_run_ids=(
    '2fd083e0ec1d484d87275b2fe53e7ce6' # base model
)


group_types=(
    'sim'
    'div'
    '21'
)

user_types=(
    'test'
    # 'train'
)

fusion_strategies=(
    # 'at_least_2_common_features'
    'average'
    # 'square_average'
    # 'topk'
    # 'max'
    'common_features'
)

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        for sae_run_id in "${sae_run_ids[@]}"
        do
            for fusion_strategy in "${fusion_strategies[@]}"
            do
                python recommend_for_groups.py --dataset EchoNest --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --note "$note"
            done
        done
    done
done