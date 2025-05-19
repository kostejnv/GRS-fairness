#!/bin/bash


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
    'div'
    'random'
)

note=sizes_acts

for sae_run_id in "${sae_run_ids[@]}"
do
    for group_type in "${group_types[@]}"
    do
        python analyze_group_embedding_intersection.py --dataset EchoNest --sae_run_id "$sae_run_id" --group_type "$group_type" --group_size 3 --user_set test --note "$note"  --topk_inference
    done
done