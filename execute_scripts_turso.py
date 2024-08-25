import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm.contrib import itertools

TEMPLATE = """#!/bin/sh
#SBATCH --job-name=RS_SimOnline
#SBATCH -M ukko
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -c 4
#SBATCH -t 2-00:00:00
#SBATCH --mem=32G
<<<HARDWARE>>>
#SBATCH --chdir=/home/hoanghu/thesis/

cd /home/hoanghu/thesis/

module purge
module load Python cuDNN

source .venv/bin/activate

date

echo
echo

<<<COMMAND>>>

echo
echo

date

"""

SBATCH_GPU = """#SBATCH -p gpu
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1"""

SBATCH_CPU = """#SBATCH -p medium
#SBATCH --constraint=intel"""

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
    # 'Pop',
    # 'SLIMElastic',
    # 'ItemKNN',
    # 'BPR',
    # 'NeuMF',
    # 'LightGCN',
]
# fmt: on
DATASETS = [
    # {"name": "ml-1m", "cutoff_date": "991854688"},
    {"name": "amazon-beauty", "cutoff_date": "1373328000"},
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

        cmd = f"srun python run_pipeline.py -m {arg_model} -d {arg_dataset} {arg_scheme} {arg_cutoff_date}"

        # Create script
        hardware = (
            SBATCH_CPU if model in ["Pop", "SLIMElastic", "Caser"] else SBATCH_GPU
        )
        script = TEMPLATE.replace("<<<HARDWARE>>>", hardware)
        script = script.replace("<<<COMMAND>>>", cmd)

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
