import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import colorlog
import numpy as np
import pandas as pd
from colorama import init
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import Dataset
from recbole.utils import init_seed
from recbole.utils.logger import RemoveColorFilter

log_colors_config = {
    "DEBUG": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}


class Paths:
    def __init__(self, model: str, dataset: str, scheme: str) -> None:
        self.model, self.dataset = model, dataset

        tag = datetime.now().strftime("%b%d_%H%M%S")
        self.path_root = Path("logs") / f"{tag}_{model}_{dataset}_{scheme}"
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

    def get_path_pretrain_ckpt(
        self,
        save_step: int = 50,
        dir_pretrained: str = "pretrained",
    ):
        file_name = f"{self.model}-{self.dataset}-{save_step}.pth"

        path = Path(dir_pretrained) / file_name
        if not path.exists():
            path = self.path_root / "ckpts" / file_name

        return path.as_posix()

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

    def get_path_dataset_conf(self):
        return (self.path_root_conf / "datasets" / f"{self.dataset}.yml").as_posix()

    def get_path_tuning_conf(self):
        return (self.path_root_conf / f"tuning_{self.model}.hyper").as_posix()


def init_logger(config, paths: Paths):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config["state"] is None or config["state"].lower() == "info":
        level = logging.INFO
    elif config["state"].lower() == "debug":
        level = logging.DEBUG
    elif config["state"].lower() == "error":
        level = logging.ERROR
    elif config["state"].lower() == "warning":
        level = logging.WARNING
    elif config["state"].lower() == "critical":
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(paths.get_path_log())
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])


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
        default="CE",
        choices=["BPR", "CE"],
    )
    parser.add_argument("-t", dest="cutoff_time", type=str, default=None)
    parser.add_argument("-s", type=str, default=None, dest="scheme")

    args = parser.parse_args()
    return args


def get_loader(
    dataset: Dataset,
    config: Config,
    separate_activeness: bool,
    cutoff: float | int | str | None,
) -> dict:
    """Get train, validation and testing loader. Create validation loader and test loader separately for active/inactive users.

    Args:
        dataset (Dataset): recbole Dataset
        config (Config): config
        separate_activeness (bool): if separate the test and val loader into active and inactive test loader
        cutoff (float | int | str | None): cutoff timestamp

    Returns:
        dict: data loaders
    """
    assert dataset.inter_feat is not None

    ARGS_NEG_SAMPLE = {"valid": "pop100", "test": "pop100"}
    ARGS_NON_NEG_SAMPLE = {"valid": "full", "test": "full"}

    val_act_ns, test_act_ns = None, None
    val_inact_ns, test_inact_ns = None, None
    val_act_non, test_act_non = None, None
    val_inact_non, test_inact_non = None, None

    if separate_activeness is True:
        assert cutoff is not None

        if not isinstance(cutoff, float):
            cutoff = float(cutoff)

        feat = dataset.inter_feat

        # Determine min/max timestamp for each user
        timestamp_byuser = feat.groupby("user_id")["timestamp"]
        min_ts = (
            timestamp_byuser.min().reset_index().rename(columns={"timestamp": "min_ts"})
        )
        max_ts = (
            timestamp_byuser.max().reset_index().rename(columns={"timestamp": "max_ts"})
        )
        user = min_ts.merge(max_ts, on="user_id", how="inner")

        # Create new dataset from features of active/inactive users
        condition_active_user = (user["min_ts"] <= cutoff) & (cutoff <= user["max_ts"])
        user_inactive = user[~condition_active_user]["user_id"]
        user_active = user[condition_active_user]["user_id"]

        feat_active = feat[feat["user_id"].isin(user_active)].copy()
        feat_inactive = feat[feat["user_id"].isin(user_inactive)].copy()

        ds_act = dataset.copy(feat_active)
        ds_inact = dataset.copy(feat_inactive)

        assert len(dataset) - len(ds_act) - len(ds_inact) == 0

        # Create active/inactive val/test dataloader for negative sampling and non-sampling cases
        config["eval_args"]["mode"] = ARGS_NEG_SAMPLE
        _, val_act_ns, test_act_ns = data_preparation(config, ds_act)
        _, val_inact_ns, test_inact_ns = data_preparation(config, ds_inact)

        config["eval_args"]["mode"] = ARGS_NON_NEG_SAMPLE
        ds_act = dataset.copy(feat_active)
        ds_inact = dataset.copy(feat_inactive)

        _, val_act_non, test_act_non = data_preparation(config, ds_act)
        _, val_inact_non, test_inact_non = data_preparation(config, ds_inact)

    # Create
    config["eval_args"]["mode"] = ARGS_NEG_SAMPLE
    ds = dataset.copy(dataset.inter_feat)
    train, val_ns, test_ns = data_preparation(config, ds)

    config["eval_args"]["mode"] = ARGS_NON_NEG_SAMPLE
    _, val_non, test_non = data_preparation(config, dataset)

    out = {
        "train": train,
        #
        "val_ns": val_ns,
        "test_ns": test_ns,
        "val_non": val_non,
        "test_non": test_non,
        #
        "val_act_ns": val_act_ns,
        "test_act_ns": test_act_ns,
        "val_inact_ns": val_inact_ns,
        "test_inact_ns": test_inact_ns,
        #
        "val_act_non": val_act_non,
        "test_act_non": test_act_non,
        "val_inact_non": val_inact_non,
        "test_inact_non": test_inact_non,
    }

    return out


def refine_result(result):
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, np.float32 | np.float64):
                result[k] = v.item()

        out = result

    elif isinstance(result, np.float32 | np.float64):
        out = result.item()

    return out
