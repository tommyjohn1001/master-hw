import argparse
import sys
import warnings
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import torch
from recbole.data import create_dataset
from recbole.utils import ModelType, get_model, get_trainer, init_seed

import src.utils as utils
from src.real_temporal import SimulatedOnlineDataset, SimulatedOnlineSequentialDataset

warnings.simplefilter(action="ignore", category=FutureWarning)

BLANK = {
    "ndcg@10": np.nan,
    "precision@10": np.nan,
    "recall@10": np.nan,
    "mrr@10": np.nan,
    "hit@10": np.nan,
    "map@10": np.nan,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", dest="path", type=str, required=False, default=None)
    parser.add_argument("-w", dest="weights", type=str, required=False, default=None)
    parser.add_argument(
        "-s",
        dest="split",
        type=str,
        required=True,
        default="full",
        choices=["full", "test", "val"],
    )

    args = parser.parse_args()
    return args


def evaluate(path: str, split: Literal["full", "val", "test"]) -> tuple:
    print(f"Evaluate with weight: {path}")

    chkpt = torch.load(path, map_location=DEVICE)

    config = chkpt["config"]
    config["device"] = DEVICE

    # NOTE: HoangLe [Sep-12]: This piece of code is hardcoded, must remove after finish
    # if config["scheme"] == "so" and config["dataset"] == "ml-1m":
    #     config["eval_args"] = {
    #         "order": "TO",
    #         "split": {"CO": "976324045"},
    #         "group_by": "user_id",
    #         "mode": "pop100",
    #     }

    print(config)

    init_seed(config["seed"], config["reproducibility"])

    # Define dataset
    if config["scheme"] == "so":
        match config["MODEL_TYPE"]:
            case ModelType.GENERAL | ModelType.CONTEXT | ModelType.TRADITIONAL:
                ds = "SimulatedOnlineDataset"
            case ModelType.SEQUENTIAL:
                ds = "SimulatedOnlineSequentialDataset"
            case _:
                print(f"model type: {config['MODEL_TYPE']}")
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

    print(f"train          : {len(loaders['train']._dataset)}")
    print(f"val_ns         : {len(loaders['val_ns']._dataset)}")
    print(f"test_ns        : {len(loaders['test_ns']._dataset)}")
    print(f"val_non        : {len(loaders['val_non']._dataset)}")
    print(f"test_non       : {len(loaders['test_non']._dataset)}")

    if loaders["val_act_ns"] is not None:
        print(f"val_act_ns     : {len(loaders['val_act_ns']._dataset)}")
        print(f"test_act_ns    : {len(loaders['test_act_ns']._dataset)}")
        print(f"val_inact_ns   : {len(loaders['val_inact_ns']._dataset)}")
        print(f"test_inact_ns  : {len(loaders['test_inact_ns']._dataset)}")

        print(f"val_act_non    : {len(loaders['val_act_non']._dataset)}")
        print(f"test_act_non   : {len(loaders['test_act_non']._dataset)}")
        print(f"val_inact_non  : {len(loaders['val_inact_non']._dataset)}")
        print(f"test_inact_non : {len(loaders['test_inact_non']._dataset)}")

    # Define model
    model_name = config["model"]
    model = get_model(model_name)(config, loaders["train"]._dataset).to(DEVICE)
    model.load_state_dict(chkpt["state_dict"])
    model.load_other_parameter(chkpt["other_parameter"])

    # Define trainer
    trainer = get_trainer(config["MODEL_TYPE"], model_name)(config, model)

    # Start evaluation
    print("== START EVALUATING  ==")

    result_val_ns = BLANK
    result_val_non = BLANK
    result_test_ns = BLANK
    result_test_non = BLANK

    result_val_act_ns = BLANK
    result_test_act_ns = BLANK
    result_val_inact_ns = BLANK
    result_test_inact_ns = BLANK

    result_val_act_non = BLANK
    result_test_act_non = BLANK
    result_val_inact_non = BLANK
    result_test_inact_non = BLANK

    if split in ["full", "val"]:
        result_val_ns = dict(
            trainer.evaluate(loaders["val_ns"], model_file=path, show_progress=True)
        )
        print(f"result_val_ns: {result_val_ns}")

        result_val_non = dict(
            trainer.evaluate(loaders["val_non"], model_file=path, show_progress=True)
        )
        print(f"result_val_non: {result_val_non}")

    if split in ["full", "test"]:
        result_test_ns = dict(
            trainer.evaluate(loaders["test_ns"], model_file=path, show_progress=True)
        )
        print(f"result_test_ns: {result_test_ns}")

        result_test_non = dict(
            trainer.evaluate(loaders["test_non"], model_file=path, show_progress=True)
        )
        print(f"result_test_non: {result_test_non}")

    if separate_activeness is True:
        if split in ["full", "val"]:
            result_val_act_ns = dict(
                trainer.evaluate(
                    loaders["val_act_ns"], model_file=path, show_progress=True
                )
            )
            print(f"result_val_act_ns: {result_val_act_ns}")

            result_val_inact_ns = dict(
                trainer.evaluate(
                    loaders["val_inact_ns"], model_file=path, show_progress=True
                )
            )
            print(f"result_val_inact_ns: {result_val_inact_ns}")

            result_val_act_non = dict(
                trainer.evaluate(
                    loaders["val_act_non"], model_file=path, show_progress=True
                )
            )
            print(f"result_val_act_non: {result_val_act_non}")

            result_val_inact_non = dict(
                trainer.evaluate(
                    loaders["val_inact_non"], model_file=path, show_progress=True
                )
            )
            print(f"result_val_inact_non: {result_val_inact_non}")

        if split in ["full", "test"]:
            result_test_act_ns = dict(
                trainer.evaluate(
                    loaders["test_act_ns"], model_file=path, show_progress=True
                )
            )
            print(f"result_test_act_ns: {result_test_act_ns}")

            result_test_inact_ns = dict(
                trainer.evaluate(
                    loaders["test_inact_ns"], model_file=path, show_progress=True
                )
            )
            print(f"result_test_inact_ns: {result_test_inact_ns}")

            result_test_act_non = dict(
                trainer.evaluate(
                    loaders["test_act_non"], model_file=path, show_progress=True
                )
            )
            print(f"result_test_act_non: {result_test_act_non}")

            result_test_inact_non = dict(
                trainer.evaluate(
                    loaders["test_inact_non"], model_file=path, show_progress=True
                )
            )
            print(f"result_test_inact_non: {result_test_inact_non}")

    ns = {
        "val": result_val_ns,
        "val_act": result_val_act_ns,
        "val_inact": result_val_inact_ns,
        "test": result_test_ns,
        "test_act": result_test_act_ns,
        "test_inact": result_test_inact_ns,
    }
    non_ns = {
        "val": result_val_non,
        "val_act": result_val_act_non,
        "val_inact": result_val_inact_non,
        "test": result_test_non,
        "test_act": result_test_act_non,
        "test_inact": result_test_inact_non,
    }

    return model_name, config["dataset"], config["scheme"], ns, non_ns


def main():
    # Get and check arguments
    args = get_args()

    if args.path is not None or args.weights is not None:
        pass
    else:
        print("Err: Either -p or -w must be specified")
        return 1

    # Curate the list of paths
    paths = []
    if args.path is not None:
        paths.append(args.path)
    else:
        # Read TXT file containing list of paths
        with open(args.weights) as f:
            paths = f.read().split()

            assert len(paths) > 0

    # Evaluate
    records_ns, records_non_ns = [], []
    for path in paths:
        model, dataset, scheme, ns, non_ns = evaluate(path, args.split)

        # Make up results
        pairs = [(ns, records_ns), (non_ns, records_non_ns)]
        for results, records in pairs:
            record = {"dataset": dataset, "model": model, "scheme": scheme}

            for tag, result in results.items():
                for k, v in result.items():
                    name = f"{tag}_{k}"

                    record[name] = v

            records.append(record)

    # Save eval results to Excel
    tag = datetime.now().strftime("%b%d-%H%M%S")

    pd.DataFrame(records_ns).to_excel(f"{tag}_negative_sampling.xlsx", index=False)
    pd.DataFrame(records_non_ns).to_excel(
        f"{tag}_non_negative_sampling.xlsx", index=False
    )


if __name__ == "__main__":
    sys.exit(main())
