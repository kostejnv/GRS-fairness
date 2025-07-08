#!/bin/bash

# variable note
base_model_run_id=0c2c7c4b7cd5427db21b9c7022ffbc18
dataset=MovieLens
note="other_models"


group_types=(
    'sim'
    'random'
    'outlier'
)

recommender_strategies=(
    'POPULAR'
    'ADD'
    'LMS'
    'GFAR'
    'EPFuzzDA'
    'MPL'
)

for group_type in "${group_types[@]}"
do
    python recommend_for_groups.py --dataset "$dataset" --recommender_strategy ELSA --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set test --base_run_id "$base_model_run_id" --note "$note" --group_set test
done

for group_type in "${group_types[@]}"
do
    python recommend_for_groups.py --dataset "$dataset" --recommender_strategy ELSA_INT --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set test --base_run_id "$base_model_run_id" --note "$note" --group_set test
done


for group_type in "${group_types[@]}"
do
    for recommender_strategy in "${recommender_strategies[@]}"
    do
        python recommend_for_groups.py --dataset "$dataset" --recommender_strategy "$recommender_strategy" --group_type "$group_type" --group_size 3 --user_set test --base_run_id "$base_model_run_id" --note "$note" --group_set test
    done
done
