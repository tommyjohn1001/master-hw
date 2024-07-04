from collections import defaultdict
from logging import getLogger

import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed

from pipeline import utils
from pipeline.real_temporal import TimeCutoffDataset


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


def main():
    use_TimeCutoff = True
    reproducible = True
    seed = 42

    # fmt: off
    config_dict = {
        # For model
        "model": "Caser",
        "embedding_size": 64,
        "n_v": 4,
        "n_h": 8,
        "reg_weight": 1e-4,
        "dropout_prob": 0.4,
        "loss_type": "CE",

        # For data
        "dataset": "ml-1m",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "use_TimeCutoff": use_TimeCutoff,

        # For training
        "epochs": 500,
        "train_batch_size": 16384,
        "eval_step": 5,
        "stopping_step": 5,
        "learning_rate": 1e-3,
        "train_neg_sample_args": None,
        # 'train_neg_sample_args': {
        #     'distribution': 'uniform',
        #     'sample_num': 1,
        #     'dynamic':  True,
        #     'candidate_num': 0
        # },

        # For evaluation
        "eval_batch_size": 16384,
        "metrics": ["NDCG", "Precision", "Recall", "MRR", "Hit", "MAP"],
        "topk": 10,
        "valid_metric": "NDCG@10",

        # Environment
        'device': 'cuda',
        'use_gpu': True,
        'gpu_id': 0,
        "checkpoint_dir": utils.get_path_dir_ckpt(),
        "show_progress": True,
        "reproducibility": reproducible,
        "seed": seed,
    }
    # fmt: on

    if use_TimeCutoff is True:
        config_dict = {
            **config_dict,
            "eval_args": {
                "order": "TO",
                "split": {"CO": "976324045"},
                "group_by": "user_id",
                "mode": "full",
            },
        }
    else:
        config_dict = {
            **config_dict,
            "eval_args": {
                "order": "TO",
                "split": {"LS": "valid_and_test"},
                "group_by": None,
                "mode": "full",
            },
        }

    config = Config(config_dict=config_dict)

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    logger = getLogger()

    # Define data related things
    if use_TimeCutoff:
        dataset = TimeCutoffDataset(config)
    else:
        dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Define model
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])

    # Define trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Start training
    logger.info(config)
    logger.info(dataset)
    logger.info(model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=True, show_progress=config["show_progress"]
    )

    print("** Validation result")
    print(f"best_valid_score: {best_valid_score:.4f}")
    for metric, val in best_valid_result.items():
        print(f"{metric:<15}: {val:.4f}")

    test_result = trainer.evaluate(test_data)

    print("** Test result")
    for metric, val in test_result.items():
        print(f"{metric:<15}: {val:.4f}")


if __name__ == "__main__":
    main()
