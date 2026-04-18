---
title: Quick Start
description: Create, inspect, query, and read your first .kdb file.
---

This guide uses generated data first, then shows the real ingest commands for HDF5, LeRobot, and RLDS.

## 1. Build

```bash
cargo build --release
export PATH="$PWD/target/release:$PATH"
```

## 2. Create A Test Database

```bash
kino create-test demo.kdb -n 20 --frames 50
```

Add a synthetic camera stream:

```bash
kino create-test demo-images.kdb -n 10 --frames 30 --images --compress 85
```

## 3. Inspect It

```bash
kino info demo.kdb
```

Expected shape:

```text
kinodb v0.1

  File:      demo.kdb
  Episodes:  20
  Frames:    1000
  Avg len:   50.0 frames/episode
```

Use `schema` when you need the frame layout, camera dimensions, byte budget, and index location:

```bash
kino schema demo-images.kdb
```

Validate before long training jobs:

```bash
kino validate demo-images.kdb --verbose
```

## 4. Query Episodes

KQL filters episode metadata:

```bash
kino query demo.kdb "success = true"
kino query demo.kdb "embodiment = 'franka' AND num_frames >= 50"
kino query demo.kdb "task CONTAINS 'pick' AND total_reward > 0.5" --limit 10
```

## 5. Read From Python

Install the bindings first:

```bash
cd crates/kinodb-py
maturin develop --release
cd ../..
```

Read one episode:

```python
import kinodb

db = kinodb.open("demo.kdb")
print(len(db), db.num_frames(), db.version())

meta = db.read_meta(0)
episode = db.read_episode(0)

print(meta)
print(episode["actions"].shape)
print(episode["states"].shape)
print(episode["rewards"].shape)
```

For image-heavy datasets where you only need low-dimensional data:

```python
episode = db.read_episode_actions_only(0)
```

## 6. Use PyTorch

```python
from kinodb.torch import KinoDataset
from torch.utils.data import DataLoader

dataset = KinoDataset(
    "demo.kdb",
    kql_filter="success = true",
    to_tensor=True,
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    actions = batch["action"]
    states = batch["state"]
    break
```

## 7. Ingest Real Data

HDF5:

```bash
kino ingest path/to/robomimic_or_libero.hdf5 \
  --format hdf5 \
  --output data.kdb \
  --embodiment franka \
  --task "open the drawer" \
  --fps 20.0
```

LeRobot:

```bash
kino ingest path/to/lerobot_dataset \
  --format lerobot \
  --output pusht.kdb \
  --max-episodes 100
```

RLDS / TFRecord:

```bash
kino ingest path/to/rlds_dataset \
  --format rlds \
  --output bridge.kdb \
  --embodiment widowx \
  --fps 3.0
```

## 8. Merge Or Mix

Physical merge:

```bash
kino merge lift.kdb pusht.kdb --output combined.kdb
kino merge lift.kdb pusht.kdb --output successful.kdb --filter "success = true"
```

Weighted training mixture:

```bash
kino mix --source lift.kdb:0.3 --source pusht.kdb:0.7 --sample 1000
```
