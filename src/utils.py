import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from recbole.utils.logger import RemoveColorFilter


class Paths:
    def __init__(self, model: str, dataset: str) -> None:
        self.model, self.dataset = model, dataset

        tag = datetime.now().strftime("%b%d_%H%M%S")
        self.path_root = Path("logs") / f"{tag}_{model}_{dataset}"
        self.path_root.mkdir(parents=True, exist_ok=True)

        self.path_root_data = Path("data")

        self.path_root_conf = Path("configs")
        self.path_root_conf.mkdir(exist_ok=True, parents=True)

    def get_path_log(self):
        return (self.path_root / "log.log").as_posix()

    def get_path_conf(self):
        return (self.path_root / "conf.yml").as_posix()

    def get_path_dir_ckpt(self):
        return (self.path_root / "ckpts").as_posix()

    def get_path_tuning_log(self):
        return (self.path_root / "tune_result.json").as_posix()

    def get_path_data_processed(self):
        path = self.path_root_data / "processed" / f"{self.dataset}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)

        return path.as_posix()

    def get_path_data_raw(self):
        path = self.path_root_data / "raw"
        path.mkdir(parents=True, exist_ok=True)

        return path.as_posix()

    def get_path_dataloader(self):
        path = self.path_root_data / "dataloader" / f"{self.model}-{self.dataset}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)

        return path.as_posix()

    def get_path_param_conf(self):
        return (self.path_root_conf / f"conf_{self.model}.yml").as_posix()

    def get_path_tuning_conf(self):
        return (self.path_root_conf / f"tuning_{self.model}.hyper").as_posix()


def get_logger(model: str, dataset: str):
    paths = Paths(model, dataset)

    fh = logging.FileHandler(paths.get_path_log())
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    logger.add(fh, colorize=False)

    logging.basicConfig(handlers=[fh], encoding="utf-8")

    return logger


def get_suitable_cutoff(ds_name: str) -> tuple:
    """Get suitable cutoff timestamp: at which there are the most active users

    Args:
        ds_name (str): dataset name

    Returns:
        tuple: suitable timestamp and the number of active users
    """

    # Get dataset without normalizing the timestamp
    config_dict = {
        "normalize_all": False,
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "train_neg_sample_args": None,
        "eval_args": {
            "order": "TO",
            "split": {"LS": "valid_and_test"},
            "group_by": None,
            "mode": "full",
        },
    }
    config = Config(
        model="NPE",
        dataset=ds_name,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    df = create_dataset(config).inter_feat.copy()

    # Create dataframe of users and corresponding first/last timestamp
    user_max_ts = df.groupby("user_id")["timestamp"].max()
    user_min_ts = df.groupby("user_id")["timestamp"].min()
    df_user = pd.DataFrame(
        {
            "max": user_max_ts,
            "min": user_min_ts,
        },
        index=user_max_ts.index,
    )

    counts = defaultdict(int)
    for ts in df_user["min"]:
        counts[ts] += 1
    for ts in df_user["max"]:
        counts[ts] -= 1

    timestamps = sorted(counts.keys())
    accum = {}

    s = 0
    for ts in timestamps:
        s += counts[ts]
        accum[ts] = s
    series = pd.Series(accum)

    suitable_ts = series.idxmax()
    max_active_user = series[suitable_ts]

    return suitable_ts, max_active_user


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", dest="model", type=str, required=True)
    parser.add_argument("-d", dest="dataset", type=str, required=True)
    parser.add_argument(
        "-l",
        dest="loss_type",
        type=str,
        required=True,
        choices=["BPR", "CE"],
    )
    parser.add_argument("-t", dest="cutoff_time", type=str, default=None)
    parser.add_argument("--use_cutoff", action="store_true", dest="use_cutoff")
    parser.add_argument("--reproducible", action="store_true", dest="reproducible")

    args = parser.parse_args()
    return args
