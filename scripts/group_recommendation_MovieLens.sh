#!/bin/bash

# variable note
note="variants"

sae_run_ids=(
'cf0be8cd67814a928d7229ffcdf162dd'
'2b5a599fc8d5402b94992cff58b93cbf'
'86d4e5f49f8444f39a29145427f716d8'
'f1736e15ef284778b2363c7cf8bbdbb5'
'f6e7e1c2a0b145869d8608d644e9cca9'
'316a7224bf7240b89258608172c562ca'
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
    # 'average'
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
                python recommend_for_groups.py --dataset MovieLens --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --note "$note" --add_interactions 50
            done
        done
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset MovieLens --recommender_strategy ELSA --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id 0c2c7c4b7cd5427db21b9c7022ffbc18 --add_interactions 50 --note "$note"
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset MovieLens --recommender_strategy ELSA_INT --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id 0c2c7c4b7cd5427db21b9c7022ffbc18 --add_interactions 50 --note "$note"
    done
done