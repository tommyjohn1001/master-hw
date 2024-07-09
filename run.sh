#!/bin/bash

unset PYTHONPATH
source thesis_venv/bin/activate

date

echo
echo

python run_pipeline.py -m NPE -d ml-1m -l CE -t 976324045 --use_cutoff --reproducible
# ml-100k 884471835
# ml-1m 976324045

echo
echo

date
