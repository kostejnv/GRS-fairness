#!/bin/bash

# variable note
note="sizes_acts"

sae_run_ids=(
'6c0bb724890748329e14df3e682e5e4c'
'904991a0628741a58cf66c60f0e37403'
'53515fd9cb554ebc90c98375d78ee3d8'
'cb615237956d42cb8f61ba51c2a262f4'
'78cf184bebc348ff965a6fb880151113'
'f8c5604ca14047e78effc318aa15d06f'
'ade4de963f26421283233ef00488ca4d'
'24e54691e96a4ce0bd4918bbfe34efcf'
'869fa42888f14edabfada151b72c5960'
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
                python recommend_for_groups.py --dataset EchoNest --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --note "$note"  --topk_inference
            done
        done
    done
done

# for user_type in "${user_types[@]}"
# do
#     for group_type in "${group_types[@]}"
#     do
#         python recommend_for_groups.py --dataset EchoNest --recommender_strategy ELSA --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id ca2a47b49d4b4cc78ad2da62a6e02123
#     done
# done

# for user_type in "${user_types[@]}"
# do
#     for group_type in "${group_types[@]}"
#     do
#         python recommend_for_groups.py --dataset EchoNest --recommender_strategy ELSA_INT --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id ca2a47b49d4b4cc78ad2da62a6e02123
#     done
# done