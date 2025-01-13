#!/bin/bash
mkdir -p data
wget -O data/LastFm1k.tar.gz http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
tar -xzf data/LastFm1k.tar.gz -C data --strip-components=1 lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv
rm -f data/LastFm1k.tar.gz
mv data/userid-timestamp-artid-artname-traid-traname.tsv data/LastFm1k.tsv
