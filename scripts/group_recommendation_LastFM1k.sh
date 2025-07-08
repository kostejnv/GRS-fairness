#!/bin/bash

# variable note
note="sizes_acts_v2"

sae_run_ids=(
'5aa4316dfa99478381d66386a470c535'
)


group_types=(
    # 'sim'
    'outlier'
    # 'random'
)

user_types=(
    # 'test'
    'train'
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
                python recommend_for_groups.py --dataset LastFM1k --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --note "$note" --group_set test
            done
        done
    done
done

# for user_type in "${user_types[@]}"
# do
#     for group_type in "${group_types[@]}"
#     do
#         python recommend_for_groups.py --dataset LastFM1k --recommender_strategy ELSA --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id 4a43996d7eec489183ad0d6b0c00d935 --add_interactions 0 --note "$note"
#     done
# done

# for user_type in "${user_types[@]}"
# do
#     for group_type in "${group_types[@]}"
#     do
#         python recommend_for_groups.py --dataset LastFM1k --recommender_strategy ELSA_INT --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id 4a43996d7eec489183ad0d6b0c00d935 --add_interactions 0 --note "$note"
#     done
# done