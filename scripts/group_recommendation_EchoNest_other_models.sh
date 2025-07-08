#!/bin/bash

# variable note
note="other_models"

base_model_run_id=a54e61982dc24b0da7b53e3b7d1d5ac8


group_types=(
    'sim'
    # 'outlier'
    # 'random'
)

user_types=(
    'test'
    # 'train'
)

recommender_strategies=(
    'ADD'
    'LMS'
    'GFAR'
    'EPFuzzDA'
    'MPL'
)


for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        for recommender_strategy in "${recommender_strategies[@]}"
        do
            python recommend_for_groups.py --dataset EchoNest --recommender_strategy "$recommender_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id "$base_model_run_id" --note "$note"
        done
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset EchoNest --recommender_strategy ELSA --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id "$base_model_run_id" --note "$note"
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset EchoNest --recommender_strategy ELSA_INT --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id "$base_model_run_id" --note "$note"
    done
done