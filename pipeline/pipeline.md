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

timestamp is encoded somewhere. where this happen ? 
    happen during loading dataset, the timestamp is 0-1 encoded.
    _normalize() did this
    => how to encode the cutoff using the same strategy
    => how to access time field before it is normalized

