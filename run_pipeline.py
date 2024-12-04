import json
import warnings
from logging import getLogger

import yaml
from recbole.config import Config
from recbole.data import create_dataset
from recbole.trainer import HyperTuning
from recbole.utils import get_model, get_trainer, init_seed

from src import utils

warnings.simplefilter(action="ignore", category=FutureWarning)


def objective_function(config_dict=None, config_file_list: list | None = None):
    config = Config(
        config_dict=config_dict,
        config_file_list=config_file_list,
    )

    match config["loss_type"]:
        case "CE":
            pass
        case "BPR":
            config["train_neg_sample_args"] = {
                "distribution": "uniform",
                "sample_num": 1,
                "alpha": 1.0,
                "dynamic": False,
                "candidate_num": 0,
            }
        case _:
            raise NotImplementedError()

    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()

    logger.info(config)

    logger.info("== START TUNNING ITERATION ==")

    # Define data related things
    dataset = create_dataset(config)

    separate_activeness = False
    loaders = utils.get_loader(
        dataset, config, separate_activeness, config["cutoff_time"]
    )

    logger.info(f"train          : {len(loaders['train']._dataset)}")
    logger.info(f"val_ns         : {len(loaders['val_ns']._dataset)}")
    logger.info(f"test_ns        : {len(loaders['test_ns']._dataset)}")
    logger.info(f"val_non        : {len(loaders['val_non']._dataset)}")
    logger.info(f"test_non       : {len(loaders['test_non']._dataset)}")

    if loaders["val_act_ns"] is not None:
        logger.info(f"val_act_ns     : {len(loaders['val_act_ns']._dataset)}")
        logger.info(f"test_act_ns    : {len(loaders['test_act_ns']._dataset)}")
        logger.info(f"val_inact_ns   : {len(loaders['val_inact_ns']._dataset)}")
        logger.info(f"test_inact_ns  : {len(loaders['test_inact_ns']._dataset)}")

        logger.info(f"val_act_non    : {len(loaders['val_act_non']._dataset)}")
        logger.info(f"test_act_non   : {len(loaders['test_act_non']._dataset)}")
        logger.info(f"val_inact_non  : {len(loaders['val_inact_non']._dataset)}")
        logger.info(f"test_inact_non : {len(loaders['test_inact_non']._dataset)}")

    # Pretrain for specific models
    model_name = config["model"]

    if model_name in ["S3Rec"]:
        logger.info("Start pre-train...")

        config["train_stage"] = "pretrain"

        model = get_model(model_name)(config, loaders["train"]._dataset).to(
            config["device"]
        )

        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.fit(loaders["train"], verbose=True, show_progress=False)

        logger.info("Finish pre-train. Start finetune")

        config["train_stage"] = "finetune"
        config["train_neg_sample_args"] = {
            "distribution": "none",
            "sample_num": "none",
            "alpha": "none",
            "dynamic": False,
            "candidate_num": 0,
        }

    # Define model
    logger.info("Define model")

    model = get_model(model_name)(config, loaders["train"]._dataset).to(
        config["device"]
    )

    # Define trainer
    logger.info("Define trainer")

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Start training
    logger.info("Start training")

    try:
        trainer.fit(loaders["train"], verbose=True, show_progress=False)
    except ValueError as e:
        if str(e) == "Training loss is nan":
            logger.error(str(e))
            pass
        else:
            raise e

    # Start evaluating
    load_best_model = model_name not in ["ItemKNN"]
    valid_metric_name = config["valid_metric"].lower()

    result_val_ns = dict(
        trainer.evaluate(
            loaders["val_ns"], load_best_model=load_best_model, show_progress=True
        )
    )
    logger.info(f"result_val_ns: {result_val_ns}")

    result_val_non = dict(
        trainer.evaluate(
            loaders["val_non"], load_best_model=load_best_model, show_progress=True
        )
    )
    logger.info(f"result_val_non: {result_val_non}")

    # Start testing
    result_test_ns = dict(
        trainer.evaluate(
            loaders["test_ns"], load_best_model=load_best_model, show_progress=True
        )
    )
    logger.info(f"result_test_ns: {result_test_ns}")

    result_test_non = dict(
        trainer.evaluate(
            loaders["test_non"], load_best_model=load_best_model, show_progress=True
        )
    )
    logger.info(f"result_test_non: {result_test_non}")

    out = {
        "model": model_name,
        "best_valid_score": utils.refine_result(result_val_ns[valid_metric_name]),
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": utils.refine_result(result_val_ns),
        "test_result": utils.refine_result(result_test_ns),
        #
        "valid_result_non": utils.refine_result(result_val_non),
        "test_result_non": utils.refine_result(result_test_non),
    }

    # Validate and test separately active and inactive users
    if separate_activeness is True:
        result_val_act_ns = dict(
            trainer.evaluate(
                loaders["val_act_ns"],
                load_best_model=load_best_model,
                show_progress=True,
            )
        )
        logger.info(f"result_val_act_ns: {result_val_act_ns}")

        result_test_act_ns = dict(
            trainer.evaluate(
                loaders["test_act_ns"],
                load_best_model=load_best_model,
                show_progress=True,
            )
        )
        logger.info(f"result_test_act_ns: {result_test_act_ns}")

        result_val_inact_ns = dict(
            trainer.evaluate(
                loaders["val_inact_ns"],
                load_best_model=load_best_model,
                show_progress=True,
            )
        )
        logger.info(f"result_val_inact_ns: {result_val_inact_ns}")

        result_test_inact_ns = dict(
            trainer.evaluate(
                loaders["test_inact_ns"],
                load_best_model=load_best_model,
                show_progress=True,
            )
        )
        logger.info(f"result_test_inact_ns: {result_test_inact_ns}")

        result_val_act_non = dict(
            trainer.evaluate(
                loaders["val_act_non"],
                load_best_model=load_best_model,
                show_progress=True,
            )
        )
        logger.info(f"result_val_act_non: {result_val_act_non}")

        result_test_act_non = dict(
            trainer.evaluate(
                loaders["test_act_non"],
                load_best_model=load_best_model,
                show_progress=True,
            )
        )
        logger.info(f"result_test_act_non: {result_test_act_non}")

        result_val_inact_non = dict(
            trainer.evaluate(
                loaders["val_inact_non"],
                load_best_model=load_best_model,
                show_progress=True,
            )
        )
        logger.info(f"result_val_inact_non: {result_val_inact_non}")

        result_test_inact_non = dict(
            trainer.evaluate(
                loaders["test_inact_non"],
                load_best_model=load_best_model,
                show_progress=True,
            )
        )
        logger.info(f"result_test_inact_non: {result_test_inact_non}")

        out = {
            **out,
            "result_val_act_ns": utils.refine_result(result_val_act_ns),
            "result_test_act_ns": utils.refine_result(result_test_act_ns),
            "result_val_inact_ns": utils.refine_result(result_val_inact_ns),
            "result_test_inact_ns": utils.refine_result(result_test_inact_ns),
            #
            "result_val_act_non": utils.refine_result(result_val_act_non),
            "result_test_act_non": utils.refine_result(result_test_act_non),
            "result_val_inact_non": utils.refine_result(result_val_inact_non),
            "result_test_inact_non": utils.refine_result(result_test_inact_non),
        }

    logger.info("== END TUNNING ITERATION ==")

    return out


def main():
    args = utils.get_args()

    assert args.scheme in ["so", "loo"]

    paths = utils.Paths(args.model, args.dataset, args.scheme)

    # Define config

    # fmt: off
    config_dict = {
        # For model
        "model": args.model,

        # For data
        "dataset": args.dataset,
        "scheme": args.scheme,
        "cutoff_time": args.cutoff_time,
        'normalize_all': False,
        'user_inter_num_interval': "[5,inf)",
        'item_inter_num_interval': "[5,inf)",

        # For training
        "epochs": 20,
        "train_batch_size": 4096,
        "eval_step": 0,
        "learning_rate": 1e-3,

        "loss_type": "CE",
        'train_neg_sample_args': None,
        
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
        "pre_model_path": paths.get_path_pretrain_ckpt(save_step=20),
        "show_progress": True,
        'save_dataset': False,
        # 'dataset_save_path': paths.get_path_data_processed(),
        'save_dataloaders': False,
        # 'dataloaders_save_path': paths.get_path_dataloader(),
    }
    # fmt: on

    if args.scheme == "so":
        config_dict["eval_args"] = {
            "order": "TO",
            "split": {"CO": args.cutoff_time},
            "group_by": "user_id",
            "mode": "pop100",
        }
    elif args.scheme == "loo":
        config_dict["eval_args"] = {
            "order": "TO",
            "split": {"LS": "valid_and_test"},
            "group_by": None,
            "mode": "pop100",
        }
    else:
        raise NotImplementedError()

    config = Config(
        config_dict=config_dict,
        config_file_list=[paths.get_path_param_conf()],
    )

    with open(paths.get_path_conf(), "w+") as f:
        yaml.dump(config.external_config_dict, f, allow_unicode=True)

    utils.init_logger(config, paths)

    logger = getLogger()

    # Start tuning
    tuning_algo = "bayes"
    early_stop = 3
    max_evals = 7

    hp = HyperTuning(
        objective_function=objective_function,
        algo=tuning_algo,
        early_stop=early_stop,
        max_evals=max_evals,
        fixed_config_file_list=[
            paths.get_path_conf(),
            paths.get_path_param_conf(),
            paths.get_path_dataset_conf(),
        ],
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
