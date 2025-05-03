#!/bin/bash
mkdir -p data
wget -O data/MovieLens.zip https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip -o data/MovieLens.zip -d data
rm -f data/MovieLens.zip
mv data/ml-32m/ratings.csv data/MovieLens.csv
# rm -rf data/ml-32m