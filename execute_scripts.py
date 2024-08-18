import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm.contrib import itertools

TEMPLATE = """#!/bin/bash

#SBATCH --job-name=RS_SimOnline
#SBATCH --account=Project_2010450
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -c 1
#SBATCH -t 3-00:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --chdir=/scratch/project_2010450/thesis/

cd /scratch/project_2010450/thesis/

unset PYTHONPATH
source thesis_venv/bin/activate

date

echo
echo

<<<COMMAND>>>

echo
echo

date

"""

PATH_DIR_SCRIPT = Path("slurm_scripts")

# fmt: off
MODELS = [
    # Sequential
    'FPMC',
    'SASRec',
    'BERT4Rec',
    'GRU4Rec',
    'S3Rec',
    'SINE',
    'LightSANs',
    'FEARec',
    'Caser',

    # General
    # 'LightGCN',
    # 'BPR',
    # 'NeuMF',
    # 'SLIMElastic',
    # 'LightGCN',
]
# fmt: on
DATASETS = [
    {"name": "ml-1m", "cutoff_date": "991854688"},
    # {"name": "amazon-beauty", "cutoff_date": "1373328000"},
    # {"name": "yelp", "cutoff_date": "1496783090"},
    # {"name": "steam", "cutoff_date": "1476576000"},
]
SCHEMES = [
    "so",
    "loo",
]


def main():
    # Clean all scripts created previously
    if PATH_DIR_SCRIPT.exists():
        logger.info(f"Removing existing directory: {str(PATH_DIR_SCRIPT)}")
        shutil.rmtree(str(PATH_DIR_SCRIPT))

    # Start creating new scripts
    PATH_DIR_SCRIPT.mkdir(exist_ok=True, parents=True)
    tag = datetime.now().strftime(r"%m%d_%H%M%S")

    for model, dataset, scheme in itertools.product(MODELS, DATASETS, SCHEMES):
        # Tailor command
        cutoff_date = dataset["cutoff_date"]

        arg_model = model
        arg_dataset = dataset["name"]
        arg_scheme = f"-s {scheme}"
        arg_cutoff_date = f"-t {cutoff_date}"

        cmd = f"python run_pipeline.py -m {arg_model} -d {arg_dataset} {arg_scheme} {arg_cutoff_date}"

        # Create script
        script = TEMPLATE.replace("<<<COMMAND>>>", cmd)

        # Store script
        script_name = f"{tag}-{arg_model}-{arg_dataset}-{scheme}.sbatch"
        path = PATH_DIR_SCRIPT / script_name

        if path.exists():
            logger.error(str(path))
            raise AttributeError()
        with open(path, "w+") as f:
            f.write(script)

    # Run script
    for path in PATH_DIR_SCRIPT.glob("*.sbatch"):
        subprocess.run(
            [
                "sbatch",
                f"{str(path)}",
            ]
        )

        logger.info(f"Triggered: {str(path)}")


if __name__ == "__main__":
    sys.exit(main())
