#!/bin/bash


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
    'div'
    'random'
)

note=sizes_acts_v2

for sae_run_id in "${sae_run_ids[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python analyze_group_embedding_intersection.py --dataset MovieLens --sae_run_id "$sae_run_id" --group_type "$group_type" --group_size 3 --user_set test --note "$note" --topk_inference
    done
done