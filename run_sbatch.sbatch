#!/bin/bash
#SBATCH --job-name=onlinesim_rs
#SBATCH --account=Project_2010450
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -c 1
#SBATCH -t 1:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --chdir=/scratch/project_2010450/thesis/

srun /scratch/project_2010450/thesis/run.sh
