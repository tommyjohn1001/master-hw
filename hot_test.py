import argparse
import sys
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from recbole.data import create_dataset
from recbole.utils import ModelType, get_model, get_trainer, init_seed

import src.utils as utils

warnings.simplefilter(action="ignore", category=FutureWarning)

BLANK = {
    "ndcg@10": np.nan,
    "precision@10": np.nan,
    "recall@10": np.nan,
    "mrr@10": np.nan,
    "hit@10": np.nan,
    "map@10": np.nan,
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", dest="path", type=str, required=True)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    device = torch.device("cuda")

    chkpt = torch.load(args.path, map_location=device)

    config = chkpt["config"]

    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()

    logger.info(config)

    # Define dataset
    if config["scheme"] == "so":
        match config["MODEL_TYPE"]:
            case ModelType.GENERAL | ModelType.CONTEXT | ModelType.TRADITIONAL:
                ds = "SimulatedOnlineDataset"
            case ModelType.SEQUENTIAL:
                ds = "SimulatedOnlineSequentialDataset"
            case _:
                logger.info(f"model type: {config['MODEL_TYPE']}")
                raise NotImplementedError()

        dataset = eval(ds)(config)
    elif config["scheme"] == "loo":
        dataset = create_dataset(config)
    else:
        raise NotImplementedError()

    separate_activeness = config["scheme"] == "loo"
    loaders = utils.get_loader(
        dataset, config, separate_activeness, config["cutoff_time"]
    )

    # Define model
    model_name = config["model"]
    model = get_model(model_name)(config, loaders["train"]._dataset).to(
        config["device"]
    )
    model.load_state_dict(chkpt["state_dict"])
    model.load_other_parameter(chkpt["other_parameter"])

    # Define trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Start evaluation
    logger.info("== START TUNNING  ==")

    result_val_ns = dict(
        trainer.evaluate(loaders["val_ns"], model_file=args.path, show_progress=True)
    )
    result_val_non = dict(
        trainer.evaluate(loaders["val_non"], model_file=args.path, show_progress=True)
    )

    result_test_ns = dict(
        trainer.evaluate(loaders["test_ns"], model_file=args.path, show_progress=True)
    )
    result_test_non = dict(
        trainer.evaluate(loaders["test_non"], model_file=args.path, show_progress=True)
    )

    logger.info(f"result_val_ns: {result_val_ns}")
    logger.info(f"result_val_non: {result_val_non}")
    logger.info(f"result_test_ns: {result_test_ns}")
    logger.info(f"result_test_non: {result_test_non}")

    if separate_activeness is True:
        result_val_act_ns = dict(
            trainer.evaluate(
                loaders["val_act_ns"], model_file=args.path, show_progress=True
            )
        )
        result_test_act_ns = dict(
            trainer.evaluate(
                loaders["test_act_ns"], model_file=args.path, show_progress=True
            )
        )
        result_val_inact_ns = dict(
            trainer.evaluate(
                loaders["val_inact_ns"], model_file=args.path, show_progress=True
            )
        )
        result_test_inact_ns = dict(
            trainer.evaluate(
                loaders["test_inact_ns"], model_file=args.path, show_progress=True
            )
        )

        result_val_act_non = dict(
            trainer.evaluate(
                loaders["val_act_non"], model_file=args.path, show_progress=True
            )
        )
        result_test_act_non = dict(
            trainer.evaluate(
                loaders["test_act_non"], model_file=args.path, show_progress=True
            )
        )
        result_val_inact_non = dict(
            trainer.evaluate(
                loaders["val_inact_non"], model_file=args.path, show_progress=True
            )
        )
        result_test_inact_non = dict(
            trainer.evaluate(
                loaders["test_inact_non"], model_file=args.path, show_progress=True
            )
        )

        logger.info(f"result_val_act_ns: {result_val_act_ns}")
        logger.info(f"result_test_act_ns: {result_test_act_ns}")
        logger.info(f"result_val_inact_ns: {result_val_inact_ns}")
        logger.info(f"result_test_inact_ns: {result_test_inact_ns}")
        logger.info(f"result_val_act_non: {result_val_act_non}")
        logger.info(f"result_test_act_non: {result_test_act_non}")
        logger.info(f"result_val_inact_non: {result_val_inact_non}")
        logger.info(f"result_test_inact_non: {result_test_inact_non}")
    else:
        result_val_act_ns = BLANK
        result_test_act_ns = BLANK
        result_val_inact_ns = BLANK
        result_test_inact_ns = BLANK

        result_val_act_non = BLANK
        result_test_act_non = BLANK
        result_val_inact_non = BLANK
        result_test_inact_non = BLANK

    ns = [
        ("val", result_val_ns),
        ("val_act", result_val_act_ns),
        ("val_inact", result_val_inact_ns),
        ("test", result_test_ns),
        ("test_act", result_test_act_ns),
        ("test_inact", result_test_inact_ns),
    ]

    non = [
        ("val", result_val_non),
        ("val_act", result_val_act_non),
        ("val_inact", result_val_inact_non),
        ("test", result_test_non),
        ("test_act", result_test_act_non),
        ("test_inact", result_test_inact_non),
    ]

    logger.info("= FOR: NON-NEGATIVE SAMPLING")

    records = {}
    for tag, result in non:
        for k, v in result.items():
            name = f"{tag}_{k}"

            records[name] = [v]

    pd.DataFrame(records).to_excel("non_negative_sampling.xlsx", index=False)

    logger.info("= FOR: NEGATIVE SAMPLING")

    records = {}
    for tag, result in ns:
        for k, v in result.items():
            name = f"{tag}_{k}"

            records[name] = [v]

    pd.DataFrame(records).to_excel("negative_sampling.xlsx", index=False)

    init_seed(config["seed"], config["reproducibility"])


if __name__ == "__main__":
    sys.exit(main())
