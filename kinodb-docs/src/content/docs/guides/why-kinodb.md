---
title: Why kinodb?
description: The robot data fragmentation problem, the GitHub issues that motivated kinodb, and the technical shape of the solution.
---

Robot learning has a data infrastructure problem hiding in plain sight. The models are increasingly general, but the datasets they train on are still split across incompatible storage systems: HDF5 for robomimic and LIBERO, Parquet plus media files for LeRobot, TFRecord for RLDS and Open X-Embodiment, Zarr or raw folders inside individual labs, and custom Python loaders around all of it.

The result is a familiar loop: every lab writes another loader, every benchmark script encodes another schema assumption, and every mixed-dataset training run becomes a pile of brittle conversion code.

## The Pain Is Real

The original kinodb blueprint started from concrete public complaints and published systems results, not an abstract dislike of file formats.

| Source | What it exposed | Why it matters |
| --- | --- | --- |
| LeRobot issue #1623 | SmolVLA training spending more time in the dataloader than backprop | Data loading can dominate wall-clock training even when the model code is fine |
| LeRobot issue #1346 | Whole-dataset memory pressure; GR00T-style fine-tuning requiring hundreds of GB for images | "Just load it into RAM" breaks down at robotics scale |
| LeRobot issue #2446 | Many LIBERO variants on the Hub across format versions | Format drift creates duplicated datasets and loader incompatibility |
| LeRobot issue #1434 | Recording-time video encoding bottlenecks | Data infrastructure affects collection, not only training |
| Robo-DM, ICRA 2025 | RLDS reported as dramatically larger per episode than necessary; LeRobot slower than optimized loading | Robotics data tooling has measurable systems overhead |
| RLDS / OXE practice | TensorFlow dependency and underdocumented conventions | PyTorch-heavy robotics stacks pay an integration tax |

These issues all point at the same missing layer: a database engine that understands trajectories as trajectories.

## Why Existing Formats Fall Short

| Format | Strength | Robotics weakness |
| --- | --- | --- |
| HDF5 | Mature hierarchical binary format with efficient array access | No trajectory query language, no standard episode schema, awkward concurrent workflows |
| LeRobot Parquet | HuggingFace-native ecosystem and tabular metadata | Parquet is optimized for column analytics, not random episode reads |
| RLDS / TFRecord | Standardized reinforcement-learning episode representation | Sequential access, TensorFlow dependency, high storage overhead in common deployments |
| Zarr | Chunked N-dimensional arrays | Useful storage primitive, but no built-in robot episode abstraction |
| Custom folders | Easy to start | No portability, no schema validation, no shared query or mixing semantics |

None of these formats are "bad." They were built for different jobs. kinodb's bet is that robot learning deserves an episode-first database layer that can ingest from all of them.

## The kinodb Idea

kinodb is an embedded trajectory database written in Rust. It stores a dataset as a single `.kdb` file with:

- a fixed-size file header,
- contiguous per-episode payloads,
- length-prefixed metadata,
- packed `f32` state/action arrays,
- optional image payloads with compressed-image pass-through,
- an end-of-file episode index for O(1) lookup.

The workflow is deliberately simple:

```bash
# HDF5: robomimic, LIBERO, DROID-style data.
kino ingest data.hdf5 --output data.kdb --format hdf5 --embodiment franka

# LeRobot v2/v3 directory.
kino ingest ./lerobot_pusht --output pusht.kdb --format lerobot

# RLDS / TFRecord directory.
kino ingest ./bridge_rlds --output bridge.kdb --format rlds --embodiment widowx
```

After ingest, every dataset has the same read path:

```bash
kino info data.kdb
kino schema data.kdb
kino query data.kdb "embodiment = 'franka' AND success = true"
```

```python
import kinodb

db = kinodb.open("data.kdb")
hits = db.query("task CONTAINS 'pick' AND num_frames > 50")
ep = db.read_episode(hits[0])
```

## What kinodb Changes

### One API after ingest

HDF5 demos, LeRobot episodes, and RLDS TFRecord steps become the same logical object: metadata plus frames. Training code no longer has to branch on `h5py`, `pyarrow`, and TensorFlow parsing rules.

### Episode-first access

Robot training typically samples episodes or windows inside episodes. A Parquet scan is excellent when the question is "read one column across millions of rows." A trajectory database wants "give me episode 817 and its metadata now." kinodb puts that access pattern in the file layout.

### Query and mixing as first-class operations

KQL exists because every robotics project eventually wants filters like:

```bash
kino query bridge.kdb "success = true AND task CONTAINS 'drawer'"
kino merge *.kdb --output successful.kdb --filter "success = true"
kino mix --source bridge.kdb:0.4 --source aloha.kdb:0.6 --sample 1000
```

Native HDF5, Parquet, and TFRecord do not provide the shared trajectory metadata semantics needed to make this uniform.

### Rust engine, Python interface

The storage and parsing paths live in Rust for predictable performance, memory-mapped I/O, and a single CLI binary. Python remains the user-facing training interface through PyO3, NumPy arrays, and PyTorch datasets.

## Current Results

The benchmark history behind the launch has three important claims:

| Claim | Evidence recorded in the benchmark history |
| --- | --- |
| Conversion preserves data | 15/15 datasets exact match; robomimic correctness issue was traced to benchmark-side lexicographic sorting and fixed |
| Metadata operations become cheap | Tabular median metadata scan speedup: 375x; image datasets: 605-2,648x |
| Episode access improves on the target workload | Tabular median random read speedup: 8.6x; image random reads: 3.3-20x once JPEG pass-through was working |

The honest framing is important: kinodb is not claiming to beat HDF5 at every raw array read. HDF5 is extremely good at direct array access. kinodb's contribution is the unified trajectory layer: indexed episodes, metadata queries, cross-format mixing, validation, and a Python training bridge.

## Current Scope vs Roadmap

Implemented now:

- `.kdb` reader/writer with memory-mapped reads.
- HDF5 ingest for robomimic/LIBERO-style `data/demo_*` files.
- LeRobot v2/v3 Parquet ingest, including action/state list columns and image struct payloads.
- RLDS TFRecord parser without a TensorFlow runtime dependency.
- KQL metadata filters.
- `info`, `schema`, `validate`, `query`, `mix`, `merge`, `export`, and `bench` CLI commands.
- PyO3 Python bindings and PyTorch dataset helpers.
- gRPC server and Python client.

Roadmap:

- Window-level frame sampling instead of whole-episode reads.
- Raw compressed image return path and lazy image decode.
- More complete video segment indexing.
- Shared-memory serving for single-node high-throughput training.
- Published wheels and packaged CLI releases.
