#!/bin/bash

# ml-1m 991854688
# amazon-digital-music 1403568000
# ml-100k 884471835

# 1. General RS: ItemKNN BPR NeuMF ENMF
python run_pipeline.py -m NeuMF -d ml-1m -t 991854688 --use_cutoff --reproducible
# python run_pipeline.py -m BPR -d amazon-digital-music --reproducible

# 2. Sequential: NPE HGN BERT4Rec GRU4Rec
# python run_pipeline.py -m HGN -l CE -d amazon-digital-music -t 1403568000 --use_cutoff --reproducible
# python run_pipeline.py -m HGN -l CE -d amazon-digital-music --reproducible

# For model: BPR

# 3. Sequential with LOO with removed inactive users

python run_pipeline.py -m NPE -d ml-1m -t 991854688 --filter-inactive --reproducible
