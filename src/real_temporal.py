import numpy as np

# from recbole.data.dataloader import *
import torch
from pandas import DataFrame
from recbole.data.dataset import Dataset, SequentialDataset
from recbole.utils import (
    FeatureType,
    set_color,
)
from torch import Tensor


class SimulatedOnlineDataset(Dataset):
    def __init__(self, config):
        self.timestamp_max, self.timestamp_min = 0.0, 0.0
        self.cutoff, self.cutoff_conv = 0.0, 0.0

        super().__init__(config)

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

        is_normalized = (
            self.config["normalize_field"]
            and self.time_field in self.config["normalize_field"]
        ) or self.config["normalize_all"]
        if is_normalized:

            def norm_timestamp(timestamp: float):
                mx, mn = self.timestamp_max, self.timestamp_min
                if mx == mn:
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

            if n_trainval <= 1:
                continue

            indices_train.extend(grouped_index[: n_trainval - 1])
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


class SimulatedOnlineSequentialDataset(SimulatedOnlineDataset, SequentialDataset):
    pass
