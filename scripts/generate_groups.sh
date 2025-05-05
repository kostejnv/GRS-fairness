#!/bin/bash

python synthetic-group-generation.py --user_set train --dataset LastFM1k --run_id 4a43996d7eec489183ad0d6b0c00d935

python synthetic-group-generation.py --user_set test --dataset EchoNest --run_id ca2a47b49d4b4cc78ad2da62a6e02123

python synthetic-group-generation.py --user_set test --dataset MovieLens --run_id 0c2c7c4b7cd5427db21b9c7022ffbc18