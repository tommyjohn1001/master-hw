Thesis

# 1. Notes with libraries

Use `pip` to install necessary libs in `requirements.txt`.

# 2. Dealing with `slumr`

This instruction is specialized for `Puhti`.

## 2.1. Install necessary libs

1. Activate Python `slurm` module with

```
module purge
module load python-data/3.10-24.04
module use /appl/soft/ai/singularity/modulefiles/
```

2. Create 'venv' and install necessary libs

```
python3 -m venv <venv_name>
source <venv_name>/bin/activate
pip install -r requirements.txt
```

## 2.2. Submit job

Run the following command to submit job

```
sbatch run_sbatch.sbatch
```

## 2.3. Some useful commands for job management

1. To check GPU utilization, run

```
seff <job_id>
```

2. To check our running tasks, run

```
squeue -u <username>
```

or

```
squeue --job <jobid>
```
