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

# 3. recbole source code modifications

By the time of this work, the latest version of recbole is `1.2` which contains tons of issues. This section describes what we modified the source code in order to make every algorithms and datasets work

1. With `LightGCN`

In file `recbole/model/general_recommender/lightgcn.py` line **113**, replace the line

```python
A._update(data_dict)
```

with

```python
for (i, j), v in data_dict.items():
    A[i, j] = v
```

2. With dataset `steam`

First, we have to download 2 processed files of this dataset from [here](link). Note that, file `steam.item` differs from the processed file `steam.item` which `recbole` team provides [here](https://drive.google.com/drive/folders/1PUsk-0rsRgea7wdeI4-vA8iRtEAUEfeK). In particular, we replace the field `id` by `product_id` to match with the field `product_id` in `steam.inter`.

In addition, in file `recbole/data/dataset/dataset.py`, line **486**, replace the following

```python
df = pd.read_csv(
    filepath,
    delimiter=field_separator,
    usecols=usecols,
    dtype=dtype,
    encoding=encoding,
    engine="python",
)
```

with

```python
if filepath.endswith("steam.item"):
    engine = None
else:
    engine = "python"
df = pd.read_csv(
    filepath,
    delimiter=field_separator,
    usecols=usecols,
    dtype=dtype,
    encoding=encoding,
    engine=engine,
)
```
