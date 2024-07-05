#!/bin/bash

unset PYTHONPATH
source thesis_venv/bin/activate

date

echo
echo

python run_pipeline.py -m Caser -d ml-1m -l BPR -t 976324045 --use_cutoff --reproducible

echo
echo

date
