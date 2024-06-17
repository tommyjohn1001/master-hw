import sys
from logging import getLogger

import numpy as np

# from recbole.data.dataloader import *
import torch
from pandas import DataFrame
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset import Dataset, SequentialDataset
from recbole.data.transform import construct_transform
from recbole.utils import (
    FeatureType,
    get_environment,
    get_flops,
    get_model,
    get_trainer,
    init_logger,
    init_seed,
    set_color,
)
from torch import Tensor


class TimeCutoffDataset(SequentialDataset):
    def __init__(self, config):
        self.timestamp_max, self.timestamp_min = 0.0, 0.0
        self.cutoff, self.cutoff_conv = 0.0, 0.0

        super().__init__(config)

    def _normalize(self):
        # Extract max-min of field self.time_field
        # feat_timestamp = self.field2feats(self.time_field)[0]
        # assert feat_timestamp and self.time_field in feat_timestamp, f"Feat not exist field '{self.time_field}'"

        # self.timestamp_max = np.max(feat_timestamp[self.time_field])
        # self.timestamp_min = np.min(feat_timestamp[self.time_field])

        self.timestamp_max = np.max(self.inter_feat[self.time_field])
        self.timestamp_min = np.min(self.inter_feat[self.time_field])

        return super()._normalize()

    def _fill_nan(self):
        """Missing value imputation.

        For fields with type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN`, missing value will be filled by
        ``[PAD]``, which indexed as 0.

        For fields with type :obj:`~recbole.utils.enum_type.FeatureType.FLOAT`, missing value will be filled by
        the average of original data.

        Note:
            This is similar to the recbole's original implementation. The difference is the change in inplace operation to suit the pandas 3.0
        """
        self.logger.debug(set_color("Filling nan", "green"))

        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            for field in feat:
                ftype = self.field2type[field]
                if ftype == FeatureType.TOKEN:
                    feat[field] = feat[field].fillna(value=0)
                elif ftype == FeatureType.FLOAT:
                    feat[field] = feat[field].fillna(value=feat[field].mean())
                else:
                    dtype = np.int64 if ftype == FeatureType.TOKEN_SEQ else np.float64
                    feat[field] = feat[field].apply(
                        lambda x: (
                            np.array([], dtype=dtype) if isinstance(x, float) else x
                        )
                    )

    def build(self):
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            super().build()

        # ordering
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args == "TO":
            self.sort(by=self.time_field)
        else:
            raise AssertionError("The ordering_method must be 'TO.")

        # splitting & grouping
        split_args = self.config["eval_args"]["split"]
        if split_args is None:
            raise ValueError("The split_args in eval_args should not be None.")
        if not isinstance(split_args, dict):
            raise ValueError(f"The split_args [{split_args}] should be a dict.")

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        if split_mode != "CO":
            raise NotImplementedError("The split_mode must be 'CO'.")
        elif split_mode == "CO":
            cutoff = split_args["CO"]
            # NOTE: HoangLe [Jun-05]: cutoff may come with different types: string, int

            group_by = self.config["eval_args"]["group_by"]
            datasets = self.split_by_cuttoff(cutoff=cutoff, group_by=group_by)

        return datasets

    def split_by_cuttoff(self, cutoff: str | int, group_by: str) -> list[Dataset]:
        """Split the interations by cutoff date

        Args:
            cutoff (str | int): cutoff date in Unix timestamp format
            group_by (str): field to group by, usually the user_id

        Returns:
            list[Dataset]: list of training/validation/testing dataset, whose interaction features has been split.

        Notes:
            cutoff may be different types: string of Unix timestamp (e.g. '1717923174'), integer of Unix timestamp (e.g. 1717923174)
        """

        self.logger.debug(f"split by cutoff date = '{cutoff}', group_by=[{group_by}]")

        assert self.inter_feat

        # Convert cutoff to suitable format and apply 0-1 normalization with max/min timestamp
        cutoff_conv = float(cutoff)
        self.cutoff = cutoff_conv

        def norm_timestamp(timestamp: float):
            mx, mn = self.timestamp_max, self.timestamp_min
            if mx == mn:
                self.logger.warning(
                    f"All the same value in [{field}] from [{feat}_feat]."
                )
                arr = 1.0
            else:
                arr = (timestamp - mn) / (mx - mn)
            return arr

        cutoff_conv = norm_timestamp(cutoff_conv)
        self.cutoff_conv = cutoff_conv

        match self.inter_feat[group_by]:
            case DataFrame():
                inter_feat_grouby_numpy = self.inter_feat[group_by].to_numpy()
            case Tensor():
                inter_feat_grouby_numpy = self.inter_feat[group_by].numpy()
            case _:
                raise TypeError(
                    f"self.inter_feat[group_by] has type: {type(self.inter_feat[group_by])} - which must be either DataFrame() or Tensor()"
                )

        grouped_inter_feat_index = self._grouped_index(inter_feat_grouby_numpy)

        indices_train, indices_val, indices_test = [], [], []
        for grouped_index in grouped_inter_feat_index:
            df_each_user = self.inter_feat[grouped_index]

            n_trainval = torch.sum(
                (df_each_user[self.time_field] <= self.cutoff_conv).to(
                    dtype=torch.int32
                )
            )
            n_test = len(df_each_user) - n_trainval

            if n_trainval == 0:
                continue

            if n_trainval >= 1:
                indices_train.extend(grouped_index[: n_trainval - 1])
            if n_trainval >= 2:
                indices_val.append(grouped_index[n_trainval - 1])
            if n_test > 0:
                indices_test.append(grouped_index[n_trainval])

        self._drop_unused_col()
        next_df = [
            self.inter_feat[index]
            for index in [indices_train, indices_val, indices_test]
        ]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds


def run_recbole_with_TimeCutoff(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = TimeCutoffDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process


def main():
    model_name = "NPE"
    dataset_name = "ml-100k"

    config_dict = {
        "use_gpu": True,
        "eval_args": {
            "order": "TO",
            "split": {"CO": "886349689"},
            "group_by": "user_id",
        },
        "train_neg_sample_args": None,
    }

    run_recbole_with_TimeCutoff(
        model=model_name, dataset=dataset_name, config_dict=config_dict
    )


if __name__ == "__main__":
    main()
