#!/bin/bash

unset PYTHONPATH
source thesis_venv/bin/activate

date

echo
echo

python run_pipeline.py

echo
echo

date
