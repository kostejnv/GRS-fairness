#!/bin/bash

# variable note
note="sizes_acts_v2"

sae_run_ids=(
'f94c2b71ea6f49ea8b7f15ec7bcd898f'
'23066f676669401ba3a51ec80ee67611'
'187b834845454facbc47846538884165'
'1401a31beaba40a4b4605dd0c8cfdd16'
'253f2ec70dec4c05be3696b2356b80fb'
'fcc282c69ba8421189f1ff7a6c23029b'
'd07c9ad6b69540778a1cd53099b91670'
'312d5703119b4d5abb359579774c26f4'
'db01455e879a446bbc53c6ef3e0ca723'
)


group_types=(
    'sim'
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
                python recommend_for_groups.py --dataset LastFM1k --sae_run_id "$sae_run_id" --use_base_model_from_sae --recommender_strategy SAE --SAE_fusion_strategy "$fusion_strategy" --group_type "$group_type" --group_size 3 --user_set "$user_type" --note "$note" --add_interactions 0 --topk_inference
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