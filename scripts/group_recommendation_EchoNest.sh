#!/bin/bash

# variable note
note="sizes"

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
    # 'common_features'
)

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        for sae_run_id in "${sae_run_ids[@]}"
        do
            for fusion_strategy in "${fusion_strategies[@]}"
            do
                python recommend_for_groups.py --dataset EchoNest --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --note "$note" --add_interactions 1000
            done
        done
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset EchoNest --recommender_strategy ELSA --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id ca2a47b49d4b4cc78ad2da62a6e02123 --add_interactions 1000
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset EchoNest --recommender_strategy ELSA_INT --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id ca2a47b49d4b4cc78ad2da62a6e02123 --add_interactions 1000
    done
done