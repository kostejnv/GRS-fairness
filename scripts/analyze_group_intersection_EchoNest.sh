#!/bin/bash

sae_run_ids=(
'93be1c1d46bc4ff3ba2c10f78038a1bd'
'5d579f5026b949cf9a2320ea66166fe9'
'5ffd2f5dc3fd43c4a9180a10fa57a1aa'
'9039861f66b0489db65e861b9a6eb993'
'7d57ea4fd48f4d3689da8767f11b2c34'
'318970e320ca49e99179493b59c2a389'
'996f822bc1494caba48ccda228b62f34'
'be68cffbcf6f456bb6a9ff34c473e035'
'dbd825b06ba6423f8c5e7b348dc3ca1a'
'53b73a30e1f34e00a98325e1f1c8e65f'
'1282a191d880446dbae7b127a9dcbeae'
'17f0e8ca2a37445bbd608e9ab1c28f3e'
'c7fd02d3f30449e087820d2cde89b63a'
'2fd083e0ec1d484d87275b2fe53e7ce6'
)

group_types=(
    'sim'
    'div'
    '21'
)

user_types=(
    'test'
    'train'
)

for user_type in "${user_types[@]}"
do
    for group_type in "${group_types[@]}"
    do
        for sae_run_id in "${sae_run_ids[@]}"
        do
            python analyze_group_embedding_intersection.py --dataset EchoNest --sae_run_id "$sae_run_id" --group_type "$group_type" --group_size 3 --user_set "$user_type"
        done
    done
done