---
title: IO Performance
description: Real-dataset IO results from the benchmark history.
---

The launch benchmark suite compared native dataset loaders against `.kdb` files after ingest. The suite covered LeRobot Parquet datasets, robomimic HDF5 datasets, and image-heavy LeRobot/LIBERO variants.

:::note
These docs record the benchmark results from the project chat history and PDF notes. The benchmark scripts/results zip are not currently committed in this repository, so treat this page as the launch report rather than a reproducible benchmark artifact. The next documentation pass should add the scripts and raw JSON outputs.
:::

## Final Scorecard

After fixing robomimic episode sorting and LeRobot image detection, the recorded final summary was:

| Dataset class | Metric | Min | Median | Max |
| --- | --- | ---: | ---: | ---: |
| Tabular, 10 datasets | Open | 1x | 113x | 146x |
| Tabular, 10 datasets | Sequential read | 0.8x | 9x | 724x |
| Tabular, 10 datasets | Random read | 0.7x | 8.6x | 843x |
| Tabular, 10 datasets | Metadata scan | 48x | 375x | 612x |
| Tabular, 10 datasets | Storage reduction | 1.1x | 5.7x | 8.7x |

Image datasets:

| Metric | Range |
| --- | ---: |
| Open | 149-420x faster |
| Random read | 3.3-20x faster |
| Metadata scan | 605-2,648x faster |
| Storage | 0.9-1.0x native size with JPEG pass-through |

Correctness:

| Check | Result |
| --- | --- |
| Datasets | 15/15 exact |
| Action max absolute diff | 0.0 on every dataset |
| robomimic issue | benchmark sorted demos lexicographically; fixed to numeric sort |
| image tag issue | reporting bug in benchmark image detector, data was present |

## Representative Dataset Table

Earlier detailed results in the chat recorded these representative values:

| Dataset | Source | Open | Sequential read | Random read | Metadata scan | Storage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LeRobot PushT | Parquet | 112x | 41x | 41x | 359x | 12.3x smaller |
| LeRobot xarm_lift | Parquet | 73x | 152x | 149x | 275x | 1.7x smaller |
| LeRobot taco_play | Parquet | 82x | 463x | 785x | 658x | 1.1x smaller |
| LeRobot ALOHA insertion | Parquet | 90x | 37x | 51x | 297x | 29x smaller |
| robomimic lift | HDF5 | 27x | 9x | 8x | 48x | 8.7x smaller |
| robomimic can | HDF5 | 23x | 4x | 4x | 54x | 7.7x smaller |

The later final scorecard is more conservative on storage medians because image datasets were re-ingested with JPEG pass-through and because some earlier image runs did not contain image payloads in `.kdb`.

## What The Numbers Mean

### Open time

Opening `.kdb` maps the file and reads the header/index. Native Parquet workflows often load or inspect table structures before episode access. This is why open-time speedups are large.

### Metadata scan

Metadata scans are the cleanest kinodb win. KQL and `read_meta()` do not decode actions or images; they read compact metadata blobs addressed by the episode index.

### Sequential and random reads

The strongest tabular wins come from episode-contiguous layout and avoiding Parquet row filtering per episode. On HDF5, wins vary because HDF5 is already efficient for direct array reads.

### Image storage

The benchmark history had a few stages:

1. Raw image storage expanded files dramatically.
2. JPEG pass-through fixed the blow-up and achieved near-native parity.
3. Recompression was discussed as a possible storage knob, but the user chose not to include that path at the time.

The docs therefore claim image storage parity, not an implemented recompression win.

## Honest Limitations

HDF5 can beat kinodb on raw image-array access because `h5py` can return large contiguous arrays efficiently and kinodb currently decodes/copies images through the Python bridge.

The current best framing:

- kinodb wins strongly on metadata, open, tabular random access, and cross-format workflows,
- image-heavy reads need lazy image loading, raw compressed-byte returns, or faster decode paths,
- storage parity for image datasets is already useful because it avoids `.kdb` raw RGB expansion.

## Reproducing A Local Synthetic Benchmark

The built-in benchmark is synthetic but useful for smoke testing:

```bash
cargo build --release
target/release/kino bench -n 500 --frames 50
target/release/kino bench -n 100 --frames 20 --images
```

It measures:

- write time,
- open time,
- sequential full-episode reads,
- deterministic random reads,
- metadata-only reads,
- KQL filters.

For publication claims, use the real benchmark suite and commit the raw JSON/HTML outputs.
