#!/bin/bash

# Datasets with 1nd cutoff selection stategy: maximizing no.active users
# ml-1m 976324045
# amazon-digital-music 1367712000

# Datasets with 2nd cutoff selection stategy: minimizing MAPE difference
# ml-1m 991854688
# amazon-digital-music 1403568000
# ml-100k 884471835

# 1. General RS: ItemKNN BPR ENMF

python run_pipeline.py -m NeuMF -d ml-1m -t 991854688 -s so
# python run_pipeline.py -m BPR -d amazon-digital-music

# 2. Sequential: NPE HGN BERT4Rec GRU4Rec

# python run_pipeline.py -m HGN -l CE -d amazon-digital-music -t 1403568000 -s so
# python run_pipeline.py -m HGN -l CE -d amazon-digital-music

# 3. Sequential with LOO with removed inactive users

python run_pipeline.py -m NPE -d ml-1m -t 991854688 --filter-inactive
