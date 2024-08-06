# Pipeline

1. _temporal_ or _ordinal_

Currently, _recbole_ implements _temporal_ by sorting the data w.r.t the timestamp (with MovieLens) and then applies leave-one-out strategy to split train/test. This is not real _temporal_

2. TimeCutoffDataset

This is a modification from class **Dataset** of `recbole`. It introduces the cutoff date used to split the train/val/test datasets.

The configuration is

```yaml
eval_args:
  group_by: user_id
  order: TO
  split: { "CO": "<something>" }
  mode: full
```

`TimeCutoffDataset` behaves as follow:

1. Select a cutoff date, the cutoff is also applied 0-1 encoding in \_normalize()
2. To split train/val/test: For each user, the cutoff date separates the interaction list (sorted by timestamp) into two halves. In the first half, all but last interaction (aka the interaction closest to the cutoff date) belong to the training split. The last interaction belongs to the validating split and the first interaction of the second half belongs to the testing set. As such, for each user, there is only one interaction of that user in the validating set and one interaction in the testing set.

Corner cases:

- If a user has interactions happened all before cutoff date, that user won't appear in test set
- If all interactions of a user are after the cutoff date, that user won't appear in neither training, validating nor testing set.

timestamp is encoded somewhere. where this happen ?
happen during loading dataset, the timestamp is 0-1 encoded.
\_normalize() did this
=> how to encode the cutoff using the same strategy
=> how to access time field before it is normalized

# 2. Dataset `TimeCutoffDataset`

## 2.1. Functional requirements

- Must be reproducible: `reproducibility` and `seed` in config
-

Choose MovieLens-100k to test the pipeline
The pipeline must be able to select any model implemented in RecBole and run list of steps

Currently, recbole implements `temporal` by sorting the interaction w.r.t. timing factor and then splitting the train/test w.r.t. the user_id => remove the group_by user_id and implement the cut-off
 Implement true temporal cut-offs on top of RecBole's temporal ordering and leave-one-out evaluation
Notes:
