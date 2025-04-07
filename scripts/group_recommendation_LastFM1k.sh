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

group_types=(
    'sim'
    'div'
    '21'
)

user_types=(
    # 'test'
    'train'
)

fusion_strategies=(
    'at_least_2_common_features'
    'average'
    'square_average'
    'topk'
    'max'
    'common_features'
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
        for sae_run_id in "${sae_run_ids[@]}"
        do
            for fusion_strategy in "${fusion_strategies[@]}"
            do
                python recommend_for_groups.py --dataset LastFM1k --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type"
            done
        done
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        for recommender_strategy in "${recommender_strategies[@]}"
        do
            python recommend_for_groups.py --dataset LastFM1k --recommender_strategy "$recommender_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type"
        done
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset LastFM1k --recommender_strategy ELSA --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type"
    done
done