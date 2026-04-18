---
title: Ingesting Data
description: Convert HDF5, LeRobot, and RLDS datasets into .kdb files.
---

`kino ingest` normalizes external robot datasets into one `.kdb` file. The current CLI supports:

- `--format hdf5`
- `--format lerobot`
- `--format rlds` or `--format tfrecord`

## Command Shape

```bash
kino ingest <SRC> \
  --format <hdf5|lerobot|rlds> \
  --output <OUT.kdb> \
  --embodiment <robot-name> \
  --task "<optional task>" \
  --fps <hz> \
  --max-episodes <N> \
  --compress <jpeg-quality>
```

`--compress` is available for HDF5 and LeRobot ingest. It JPEG-compresses raw image frames, while already-compressed JPEG/PNG payloads are passed through.

## HDF5

The HDF5 ingester targets robomimic/LIBERO-style files:

```text
data/
  demo_0/
    actions                 (T, action_dim) float32
    rewards                 (T,) float32       optional
    dones                   (T,) float/int     optional
    obs/
      agentview_image       (T, H, W, C) uint8 optional camera
      robot0_eef_pos        (T, D) float32     optional state
      robot0_eef_quat       (T, D) float32     optional state
      robot0_gripper_qpos   (T, D) float32     optional state
```

Example:

```bash
kino ingest lift.hdf5 \
  --format hdf5 \
  --output lift.kdb \
  --embodiment franka \
  --task "lift the cube" \
  --fps 20.0 \
  --compress 85
```

Discovery rules:

| Data | Rule |
| --- | --- |
| Episodes | Groups named `demo_*` under `data/`, sorted numerically |
| Actions | Required `actions` dataset |
| State | Any `obs/*` dataset with 2 dimensions, concatenated in sorted key order |
| Cameras | Any `obs/*` dataset with 4 dimensions |
| Success | Last reward greater than zero when rewards exist |
| Terminal | `dones[t] > 0.5`, otherwise the final frame |

The numeric sort matters. A benchmark correctness issue was traced to native comparison code sorting `demo_10` before `demo_2`; kinodb sorts episode keys numerically.

## LeRobot

The LeRobot ingester supports v2/v3-style directories with `meta/`, `data/`, and optionally `videos/`.

```text
dataset/
  meta/
    info.json
    tasks.jsonl or tasks.parquet
  data/
    chunk-000/
      file-000.parquet
  videos/
```

Example:

```bash
kino ingest ./lerobot_pusht \
  --format lerobot \
  --output pusht.kdb
```

With overrides:

```bash
kino ingest ./aloha_sim \
  --format lerobot \
  --output aloha.kdb \
  --embodiment aloha \
  --task "insert the peg" \
  --max-episodes 100 \
  --compress 85
```

Discovery rules:

| Data | Rule |
| --- | --- |
| FPS | `meta/info.json` field `fps`, default `30.0` |
| Embodiment | `meta/info.json` field `robot_type`, unless overridden |
| Tasks | `meta/tasks.jsonl` or `meta/tasks.parquet` |
| Episodes | Rows grouped by `episode_index` |
| Actions | Column named `action` or columns prefixed with `action.` |
| State | Column named `observation.state` or prefixed with `observation.state.` |
| Images | Struct columns containing image bytes, stored as `ImageObs` |

LeRobot list columns are handled. This matters because common datasets store actions/states as FixedSizeList/List arrays rather than scalar float columns.

## RLDS / TFRecord

The RLDS ingester parses TFRecord files directly rather than importing TensorFlow.

```bash
kino ingest ./bridge_rlds \
  --format rlds \
  --output bridge.kdb \
  --embodiment widowx \
  --fps 3.0
```

Expected layout:

```text
dataset/
  1.0.0/
    dataset_info.json
    train.tfrecord-00000-of-00005
    train.tfrecord-00001-of-00005
```

Parsing rules:

| Data | Rule |
| --- | --- |
| Episode boundaries | `is_first`, `is_last`, and `is_terminal` flags |
| Actions | `action` float list |
| State | Sorted `observation/*` float fields excluding image keys |
| Language | `language_instruction` or `observation/language_instruction` |
| Images | `observation/*image*` byte features decoded to RGB when possible |
| Reward | Per-step `reward`, summed into episode metadata |

## Image Storage

The writer has two paths:

| Input | Storage behavior |
| --- | --- |
| Raw RGB bytes and `--compress Q` | JPEG encode at quality `Q` |
| JPEG/PNG bytes | Pass through the compressed payload |
| Raw RGB bytes without compression | Store raw pixels |

The benchmark history found that JPEG pass-through is the right current default for LeRobot image datasets: it keeps `.kdb` near native size rather than expanding compressed images into raw RGB.

## After Ingest

Always inspect, validate, and query:

```bash
kino info data.kdb
kino schema data.kdb
kino validate data.kdb --verbose
kino query data.kdb "num_frames > 50"
```

For archival or release workflows:

```bash
kino export data.kdb --format numpy --output export/
kino export data.kdb --format json --output metadata/
```
