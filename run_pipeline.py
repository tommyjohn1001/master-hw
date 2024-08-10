import json
import warnings
from logging import getLogger

import yaml
from recbole.config import Config
from recbole.data import create_dataset
from recbole.trainer import HyperTuning
from recbole.utils import ModelType, get_model, get_trainer, init_seed

from src import utils
from src.real_temporal import SimulatedOnlineDataset, SimulatedOnlineSequentialDataset

warnings.simplefilter(action="ignore", category=FutureWarning)


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
        match config["MODEL_TYPE"]:
            case ModelType.GENERAL | ModelType.TRADITIONAL:
                dataset = SimulatedOnlineDataset(config)
            case ModelType.SEQUENTIAL:
                dataset = SimulatedOnlineSequentialDataset(config)

    else:
        dataset = create_dataset(config)

    separate_activeness = config["use_cutoff"] is False
    dataloaders = utils.get_loader(
        dataset, config, separate_activeness, config["cutoff_time"]
    )

    train_data = dataloaders["train_data"]
    valid_data = dataloaders["valid_data"]
    test_data = dataloaders["test_data"]
    valid_data_inactive = dataloaders["valid_data_inactive"]
    valid_data_active = dataloaders["valid_data_active"]
    test_data_inactive = dataloaders["test_data_inactive"]
    test_data_active = dataloaders["test_data_active"]

    logger.info(f"train_dataset         : {len(train_data._dataset)}")
    logger.info(f"valid_dataset         : {len(valid_data._dataset)}")
    logger.info(f"test_dataset          : {len(test_data._dataset)}")
    if valid_data_inactive is not None:
        logger.info(f"test_dataset_inactive : {len(test_data_inactive._dataset)}")
        logger.info(f"test_dataset_active   : {len(test_data_active._dataset)}")
        logger.info(f"valid_dataset_inactive: {len(valid_data_inactive._dataset)}")
        logger.info(f"valid_dataset_active  : {len(valid_data_active._dataset)}")

    # Define model
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])

    # Define trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Start training
    try:
        trainer.fit(train_data, valid_data, verbose=True)
    except ValueError as e:
        if str(e) == "Training loss is nan":
            pass
        else:
            raise e

    # Start evaluating
    load_best_model = model_name not in ["ItemKNN"]
    valid_metric_name = config["valid_metric"].lower()

    valid_result = dict(trainer.evaluate(valid_data, load_best_model=load_best_model))

    logger.info(f"valid_result: {valid_result}")

    # Start testing
    test_result = dict(trainer.evaluate(test_data, load_best_model=load_best_model))

    logger.info(f"test_result: {test_result}")

    out = {
        "model": model_name,
        "best_valid_score": utils.refine_result(valid_result[valid_metric_name]),
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": utils.refine_result(valid_result),
        "test_result": utils.refine_result(test_result),
    }

    # Validate and test separately active and inactive users
    if config["use_cutoff"] is False:
        assert test_data_inactive is not None
        assert test_data_active is not None
        assert valid_data_inactive is not None
        assert valid_data_active is not None

        valid_result_inactive = dict(
            trainer.evaluate(valid_data_inactive, load_best_model=load_best_model)
        )
        valid_result_active = dict(
            trainer.evaluate(valid_data_active, load_best_model=load_best_model)
        )

        test_result_inactive = dict(
            trainer.evaluate(test_data_inactive, load_best_model=load_best_model)
        )
        test_result_active = dict(
            trainer.evaluate(test_data_active, load_best_model=load_best_model)
        )

        out = {
            **out,
            "valid_result_inactive": utils.refine_result(valid_result_inactive),
            "valid_result_active": utils.refine_result(valid_result_active),
            "test_result_inactive": utils.refine_result(test_result_inactive),
            "test_result_active": utils.refine_result(test_result_active),
        }

    logger.info("== END TUNNING ITERATION ==")

    return out


def main():
    args = utils.get_args()
    paths = utils.Paths(args.model, args.dataset, args.use_cutoff)

    # Define config

    # fmt: off
    config_dict = {
        # For model
        "model": args.model,

        # For data
        "dataset": args.dataset,
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "use_cutoff": args.use_cutoff,
        "cutoff_time": args.cutoff_time,
        'normalize_all': False,
        'user_inter_num_interval': "[10,inf)",

        # For training
        "epochs": 60,
        "train_batch_size": 4096,
        "eval_step": 5,
        "stopping_step": 5,
        "learning_rate": 1e-3,
        
        # For evaluation
        "eval_batch_size": 4096,
        "metrics": ["NDCG", "Precision", "Recall", "MRR", "Hit", "MAP"],
        "topk": 10,
        "valid_metric": "NDCG@10",

        # Environment
        'gpu_id': 0,
        "seed": 42,
        "reproducibility": True,
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
            "mode": "pop100",
        }
    else:
        config_dict["eval_args"] = {
            "order": "TO",
            "split": {"LS": "valid_and_test"},
            "group_by": None,
            "mode": "pop100",
        }

    if args.loss_type is not None:
        config_dict["loss_type"] = args.loss_type

    if args.loss_type is None or args.loss_type == "CE":
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

    # Start tuning
    tuning_algo = "bayes"
    early_stop = 5
    max_evals = 15

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
    logger.info("best params: ")
    logger.info(hp.best_params)

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
