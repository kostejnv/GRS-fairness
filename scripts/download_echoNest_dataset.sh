#!/bin/bash
mkdir -p data
wget -O data/EchoNest.zip http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip
unzip -o data/EchoNest.zip -d data
rm -f data/EchoNest.zip
mv data/train_triplets.txt data/EchoNest.txt