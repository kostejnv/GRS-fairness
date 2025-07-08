#!/bin/bash

run_id=0c2c7c4b7cd5427db21b9c7022ffbc18 # Add your ELSA run ID here
dataset=MovieLens

python generate-groups.py --user_set test --dataset "$dataset" --run_id "$run_id"