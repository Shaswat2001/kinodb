---
title: PyTorch Training
description: Use kinodb from Python and PyTorch.
---

The Python bindings expose a Rust-backed `Database` plus PyTorch helpers in `kinodb.torch`.

## Install

```bash
cd crates/kinodb-py
maturin develop --release
cd ../..

python -m pip install torch numpy
```

## Database API

```python
import kinodb

db = kinodb.open("data.kdb")

print(db.num_episodes())
print(db.num_frames())
print(db.version())
print(db.summary())

meta = db.read_meta(0)
episode = db.read_episode(0)
```

`read_episode()` returns:

| Key | Type |
| --- | --- |
| `meta` | dict |
| `actions` | NumPy `float32`, shape `(T, action_dim)` |
| `states` | NumPy `float32`, shape `(T, state_dim)` |
| `rewards` | NumPy `float32`, shape `(T,)` |
| `is_terminal` | list of bool |
| `images` | dict of camera name to NumPy `uint8`, shape `(T, H, W, C)` |

For low-dimensional training on image-heavy datasets:

```python
episode = db.read_episode_actions_only(0)
```

This skips image decoding and transfer.

## Map-Style Dataset

`KinoDataset` is an alias for `KinoMapDataset`.

```python
from kinodb.torch import KinoDataset
from torch.utils.data import DataLoader

dataset = KinoDataset(
    "data.kdb",
    kql_filter="success = true",
    image_key="front",
    image_size=(224, 224),
    to_tensor=True,
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

Sample keys:

| Key | Shape |
| --- | --- |
| `action` | `(T, action_dim)` |
| `state` | `(T, state_dim)` |
| `reward` | `(T,)` |
| `image` | `(T, C, H, W)` when images exist |
| `task` | string |
| `embodiment` | string |
| `success` | bool or `None` |
| `episode_id` | int |

## Iterable Dataset

Use `KinoIterDataset` for streaming or multi-source mixtures:

```python
from kinodb.torch import KinoIterDataset
from torch.utils.data import DataLoader

dataset = KinoIterDataset(
    "data.kdb",
    kql_filter="num_frames > 50",
    shuffle=True,
    seed=42,
)

loader = DataLoader(dataset, batch_size=4)
```

## Weighted Mixtures

```python
from kinodb.torch import from_mixture
from torch.utils.data import DataLoader

dataset = from_mixture(
    {
        "robomimic_lift.kdb": 0.3,
        "pusht.kdb": 0.4,
        "aloha.kdb": 0.3,
    },
    seed=42,
    image_size=(224, 224),
)

loader = DataLoader(dataset, batch_size=4)
```

Weights are relative. They do not need to sum to one.

## Images

The current helper picks one camera:

```python
dataset = KinoDataset("data.kdb", image_key="agentview_image")
```

If `image_key` is omitted, the first camera in the episode image dictionary is used.

Resize is nearest-neighbor and dependency-free:

```python
dataset = KinoDataset("data.kdb", image_size=(128, 128))
```

The returned image tensor is normalized to `[0, 1]` and transposed to `(T, C, H, W)`.

## Multi-Worker Notes

The current `KinoMapDataset` opens the database in `__init__`. For multiprocessing-heavy workloads, prefer one of these patterns:

```python
# Simple path: single-process or low worker count.
loader = DataLoader(KinoDataset("data.kdb"), batch_size=4, num_workers=0)
```

For robust multi-worker training, use a dataset wrapper that stores the path and opens `kinodb.open()` lazily inside each worker. This pattern came up in the experiment work because database handles are not always pickle-safe across worker processes.

```python
class LazyKinoDataset:
    def __init__(self, path):
        self.path = path
        self.db = None

    def _reader(self):
        if self.db is None:
            import kinodb
            self.db = kinodb.open(self.path)
        return self.db

    def __len__(self):
        return self._reader().num_episodes()

    def __getitem__(self, idx):
        return self._reader().read_episode_actions_only(idx)
```

## Benchmark Lesson

The training benchmark scripts originally measured the wrong thing by preloading both native and `.kdb` data into NumPy arrays before timing training. Once everything is in memory, the data source no longer matters. The corrected benchmark measures disk access inside the training/inference loop.

That distinction is the launch lesson: kinodb's advantage shows up when the benchmark includes the data loading path that real large-scale training has to pay.
