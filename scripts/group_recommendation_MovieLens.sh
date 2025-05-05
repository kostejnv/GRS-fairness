#!/bin/bash

# variable note
note="sizes"

sae_run_ids=(
'3e17985c2d934ce2aec3ee8ead301a86'
'86677d0d62874cba87c4bd388dece9d0'
'1f64d72748614316b9039d3b15936032'
'a4274731b4814a74a183deeb37d87ba8'
'40a86da42e764d53a1178334c4a1b774'
'73760d7e0a7e4e5db46e98873536630e'
'2580099610b543d5bcb3f6326c063627'
'ce4f7fc2057e489bb8a0edd604d693be'
'b3912a66af1e4541bf24e10a8dc8b831'
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
    'topk'
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
                python recommend_for_groups.py --dataset MovieLens --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --note "$note" --add_interactions 50
            done
        done
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset MovieLens --recommender_strategy ELSA --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id 0c2c7c4b7cd5427db21b9c7022ffbc18 --add_interactions 50
    done
done

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python recommend_for_groups.py --dataset MovieLens --recommender_strategy ELSA_INT --SAE_fusion_strategy average --group_type "$group_type" --group_size 3 --user_set "$user_type" --base_run_id 0c2c7c4b7cd5427db21b9c7022ffbc18 --add_interactions 50
    done
done