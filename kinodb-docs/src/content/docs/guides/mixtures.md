---
title: Dataset Mixtures
description: Merge or sample weighted datasets across original source formats.
---

Modern VLA training often mixes demonstrations from different robots, tasks, and source formats. kinodb gives you two paths:

- `kino merge` creates one physical `.kdb` file.
- `kino mix` and `from_mixture()` create weighted virtual mixtures.

## Physical Merge

Use merge when you want a single file to ship, archive, upload, or train sequentially.

```bash
kino merge lift.kdb pusht.kdb aloha.kdb --output combined.kdb
```

Filter while merging:

```bash
kino merge lift.kdb pusht.kdb \
  --output successful.kdb \
  --filter "success = true"
```

This reads each input episode, applies the optional KQL filter, then writes matching episodes to a new database.

## Weighted Mixture CLI

Use mix when you want training-time sampling proportions.

```bash
kino mix \
  --source bridge.kdb:0.4 \
  --source aloha.kdb:0.3 \
  --source libero.kdb:0.3
```

Sample a distribution:

```bash
kino mix \
  --source bridge.kdb:0.4 \
  --source aloha.kdb:0.3 \
  --source libero.kdb:0.3 \
  --sample 1000 \
  --seed 42
```

Weights are relative. `4:3:3` and `0.4:0.3:0.3` are equivalent.

## Python Mixtures

```python
from kinodb.torch import from_mixture
from torch.utils.data import DataLoader

dataset = from_mixture(
    {
        "bridge.kdb": 0.4,
        "aloha.kdb": 0.3,
        "libero.kdb": 0.3,
    },
    seed=42,
    image_size=(224, 224),
)

loader = DataLoader(dataset, batch_size=8)
```

Each source can come from a different original format. The training code only sees `.kdb`.

## Rust API

```rust
use kinodb_core::Mixture;

let mut mix = Mixture::builder()
    .add("bridge.kdb", 0.4)
    .add("aloha.kdb", 0.3)
    .add("libero.kdb", 0.3)
    .seed(42)
    .build()?;

let episode = mix.sample()?;
let global_episode = mix.read_global(10)?;
let order = mix.weighted_epoch(1000);
```

## Mixed Schema Reality

Different datasets can have different action and state dimensions. The experiment history hit this directly when mixing PushT (`action_dim = 2`) with ALOHA (`action_dim = 14`): raw `torch.stack` fails unless the collate function pads or batches by schema.

Common strategies:

| Strategy | When to use |
| --- | --- |
| Pad state/action vectors to max dimension | One model with source-aware masks |
| Bucket by schema | Multi-embodiment training with separate heads |
| Train separate adapters | Different embodiments have genuinely different action spaces |
| Physical merge only same-schema data | Simplest archival/distribution path |

kinodb preserves dimensions and metadata; it does not hide schema differences from your model.

## When To Merge vs Mix

| Use case | Pick |
| --- | --- |
| Publish one converted dataset | `kino merge` |
| Build a filtered release split | `kino merge --filter` |
| Match OpenVLA-style source proportions | `kino mix` or `from_mixture()` |
| Change ratios between runs | virtual mixture |
| Maximize sequential reads from one file | physical merge |
