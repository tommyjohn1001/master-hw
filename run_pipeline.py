import json
from logging import getLogger

import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import HyperTuning
from recbole.utils import get_model, get_trainer, init_seed

from src import utils
from src.real_temporal import TimeCutoffDataset


def objective_function(config_dict=None, config_file_list=None):
    config = Config(
        config_dict=config_dict,
        config_file_list=config_file_list,
    )

    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()

    logger.info("== START TUNNING ITERATION ==")

    # Define data related things
    if config["use_cutoff"] is True:
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
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=True
    )

    # Start evaluating
    test_result = trainer.evaluate(test_data)

    logger.info("== END TUNNING ITERATION ==")

    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def main():
    seed = 42

    args = utils.get_args()
    paths = utils.Paths(args.model, args.dataset)

    # Define config

    # fmt: off
    config_dict = {
        # For model
        "model": args.model,
        "loss_type": args.loss_type,

        # For data
        "dataset": args.dataset,
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "use_cutoff": args.use_cutoff,

        # For training
        "epochs": 8,
        "train_batch_size": 4096,
        "eval_step": 1,
        "stopping_step": 3,
        "learning_rate": 1e-3,
        
        # For evaluation
        "eval_batch_size": 4096,
        "metrics": ["NDCG", "Precision", "Recall", "MRR", "Hit", "MAP"],
        "topk": 10,
        "valid_metric": "NDCG@10",

        # Environment
        'gpu_id': 0,
        "seed": seed,
        "reproducibility": args.reproducible,
        'device': 'cuda',
        'use_gpu': True,
        'data_path': paths.get_path_data_raw(),
        "checkpoint_dir": paths.get_path_dir_ckpt(),
        "show_progress": True,
        'save_dataset': True,
        'dataset_save_path': paths.get_path_data_processed(),
        'save_dataloaders': True,
        'dataloaders_save_path': paths.get_path_dataloader(),
    }
    # fmt: on

    if args.use_cutoff is True:
        config_dict["eval_args"] = {
            "order": "TO",
            "split": {"CO": args.cutoff_time},
            "group_by": "user_id",
            "mode": "full",
        }
    else:
        config_dict["eval_args"] = {
            "order": "TO",
            "split": {"LS": "valid_and_test"},
            "group_by": None,
            "mode": "full",
        }

    if args.loss_type == "CE":
        config_dict["train_neg_sample_args"] = None
    else:
        config_dict["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": 1,
            # "dynamic": True,
            # "candidate_num": 0,
        }

    config = Config(
        config_dict=config_dict,
        config_file_list=[paths.get_path_param_conf()],
    )

    with open(paths.get_path_conf(), "w+") as f:
        yaml.dump(config.external_config_dict, f, allow_unicode=True)

    init_seed(config["seed"], config["reproducibility"])
    utils.init_logger(config, paths)

    logger = getLogger()
    logger.info(config)

    ## Start tuning
    tuning_algo = "bayes"
    early_stop = 3
    max_evals = 5

    hp = HyperTuning(
        objective_function=objective_function,
        algo=tuning_algo,
        early_stop=early_stop,
        max_evals=max_evals,
        fixed_config_file_list=[paths.get_path_conf(), paths.get_path_param_conf()],
        params_file=paths.get_path_tuning_conf(),
    )
    hp.run()

    # print best parameters
    logger.info("best params: ", hp.best_params)

    # print best result
    logger.info("best result: ")
    logger.info(hp.params2result[hp.params2str(hp.best_params)])

    # export to JSON file
    tune_result = {
        "best_params": hp.best_params,
        "best_result": hp.params2result[hp.params2str(hp.best_params)],
    }
    with open(paths.get_path_tuning_log(), "w+") as f:
        json.dump(tune_result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
