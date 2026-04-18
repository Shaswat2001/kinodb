---
title: CLI Commands
description: Reference for kino and kino-serve.
---

The CLI binary is `kino`. The gRPC server binary is `kino-serve`.

```bash
kino <COMMAND>
```

## `kino create-test`

Generate synthetic `.kdb` files for smoke tests and benchmarks.

```bash
kino create-test [PATH] \
  -n <NUM_EPISODES> \
  --frames <FRAMES> \
  --images \
  --compress <QUALITY>
```

| Option | Default | Description |
| --- | --- | --- |
| `PATH` | `test.kdb` | Output path |
| `-n, --num-episodes` | `10` | Number of episodes |
| `--frames` | `50` | Frames per episode |
| `--images` | false | Add fake 64x64 RGB camera frames |
| `--compress` | none | JPEG quality `1-100` |

## `kino ingest`

Convert external datasets into `.kdb`.

```bash
kino ingest <SRC> \
  --output output.kdb \
  --format hdf5 \
  --embodiment franka \
  --task "open drawer" \
  --fps 20.0 \
  --max-episodes 100 \
  --compress 85
```

| Option | Default | Description |
| --- | --- | --- |
| `SRC` | required | Source file or directory |
| `-o, --output` | `output.kdb` | Output `.kdb` path |
| `-F, --format` | `hdf5` | `hdf5`, `lerobot`, `rlds`, or `tfrecord` |
| `-e, --embodiment` | `unknown` | Robot embodiment |
| `-t, --task` | inferred when possible | Task description |
| `--fps` | `10.0` | Control frequency |
| `--max-episodes` | none | Limit ingest |
| `--compress` | none | JPEG compression quality |

## `kino info`

Print a database summary.

```bash
kino info data.kdb
kino info data.kdb --episodes
```

| Option | Description |
| --- | --- |
| `--episodes` | Include per-episode table |

## `kino schema`

Print structural information, including file version, index offset, field dimensions, cameras, and byte budget.

```bash
kino schema data.kdb
```

Use this when you need to answer "what is inside this file?" before training.

## `kino validate`

Check file consistency and decode every episode.

```bash
kino validate data.kdb
kino validate data.kdb --verbose
```

Validation checks include:

- header/index episode count consistency,
- total frame count consistency,
- metadata decode,
- full episode decode,
- action/state dimensions,
- NaN/Inf warnings,
- image shape consistency,
- terminal-frame warning.

## `kino query`

Run a KQL filter.

```bash
kino query data.kdb "success = true AND num_frames > 50"
kino query data.kdb "task CONTAINS 'pick'" --limit 20
```

| Option | Description |
| --- | --- |
| `--limit` | Maximum results to print |

## `kino mix`

Inspect weighted virtual mixtures.

```bash
kino mix --source bridge.kdb:0.4 --source aloha.kdb:0.6
kino mix --source bridge.kdb:0.4 --source aloha.kdb:0.6 --sample 1000
```

| Option | Default | Description |
| --- | --- | --- |
| `-s, --source` | required | `path:weight`, repeatable |
| `--seed` | `42` | Sampling seed |
| `--sample` | none | Sample N episodes and print distribution |

## `kino merge`

Write multiple `.kdb` inputs into one output.

```bash
kino merge lift.kdb pusht.kdb --output combined.kdb
kino merge lift.kdb pusht.kdb --output successful.kdb --filter "success = true"
```

| Option | Default | Description |
| --- | --- | --- |
| inputs | required | Input `.kdb` files |
| `-o, --output` | `merged.kdb` | Output path |
| `-F, --filter` | none | Optional KQL filter |

## `kino export`

Export a `.kdb` file back to simple interchange formats.

```bash
kino export data.kdb --output export/ --format numpy
kino export data.kdb --output metadata/ --format json
```

| Format | Output |
| --- | --- |
| `numpy` | Episode directories with `actions.bin`, `states.bin`, `rewards.bin`, camera `.bin` files, and JSON metadata |
| `json` | Metadata-only JSON |

## `kino bench`

Run a synthetic benchmark for write, open, sequential read, random read, metadata read, and KQL.

```bash
kino bench -n 500 --frames 50
kino bench -n 100 --frames 20 --images
```

The benchmark writes to `/tmp/kinodb_bench.kdb` and removes it at the end.

## `kino-serve`

Start a gRPC server.

```bash
kino-serve data.kdb --port 50051

kino-serve \
  --source bridge.kdb:0.4 \
  --source aloha.kdb:0.6 \
  --port 50051
```

| Option | Default | Description |
| --- | --- | --- |
| file | none | Single `.kdb` file |
| `-s, --source` | none | Weighted source, repeatable |
| `-p, --port` | `50051` | Listen port |
| `--bind` | `0.0.0.0` | Bind address |
| `--seed` | `42` | Mixture sampling seed |
