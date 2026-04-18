---
title: Correctness
description: Data preservation checks and benchmark correctness lessons.
---

Correctness matters more than speed. A trajectory database is not useful if it silently changes actions, states, frame counts, or task labels.

## Recorded Final Result

The final benchmark history reports:

| Check | Result |
| --- | --- |
| Datasets checked | 15 |
| Exact datasets | 15/15 |
| `action_max_abs_diff` | `0.0` on every dataset |
| robomimic issue | fixed by numeric demo sorting in the benchmark |
| Image payload status | present after image-ingest fixes; one remaining "Images: no" label was a reporting bug |

## What Was Compared

For each dataset, the benchmark compared native reads against `.kdb` reads for sampled episodes.

| Field | Comparison |
| --- | --- |
| Actions | elementwise max absolute difference |
| States | elementwise max absolute difference when available |
| Episode length | native frame count vs `.kdb` frame count |
| Images | payload presence, shape, and camera accounting |
| Metadata | task, embodiment, action dimension, frame count |

## The robomimic Sorting Bug

The biggest correctness scare was robomimic reporting `inf` action differences. The root cause was not corrupted data. It was ordering:

```text
lexicographic: demo_0, demo_1, demo_10, demo_100, demo_2
numeric:       demo_0, demo_1, demo_2,  demo_3,   demo_4
```

kinodb ingests HDF5 demo groups in numeric order. The benchmark originally compared against native HDF5 groups in lexicographic order, so it was comparing different episodes after the first few demos.

Fix: sort demo keys by the integer after `demo_`.

## HDF5 State Semantics

HDF5 observation groups often contain many low-dimensional state keys:

```text
obs/
  robot0_eef_pos
  robot0_eef_quat
  robot0_gripper_qpos
  object
```

kinodb concatenates all 2D state keys in sorted order. Native benchmark code must do the same to compare state vectors fairly.

## Image Correctness

For image datasets, the storage story changed during development:

1. LeRobot image struct columns were initially skipped.
2. Image extraction was added.
3. Raw RGB storage caused huge files.
4. JPEG/PNG pass-through fixed storage parity.
5. The benchmark image detector still had a reporting issue in one summary, even though data was present.

Current reader behavior:

- raw image payloads are returned as raw bytes,
- compressed JPEG/PNG payloads are decoded to raw RGB on `read_episode()`,
- decode failures skip that frame image rather than crashing the entire episode read.

## Validation Command

Use the CLI validator before expensive training:

```bash
kino validate data.kdb
kino validate data.kdb --verbose
```

It checks:

- header and index parse,
- header episode count vs index length,
- total frame count vs index entries,
- per-episode metadata decode,
- full episode decode,
- action dimension consistency,
- state dimension consistency across frames,
- NaN/Inf warnings,
- image byte length vs dimensions,
- terminal-frame warning.

## What To Commit For A Paper

The launch docs preserve the conclusions, but a paper-ready repo should commit:

- raw benchmark JSON,
- exact dataset versions and HuggingFace revisions,
- native loader code,
- `.kdb` ingest commands,
- correctness comparison code,
- environment metadata,
- generated tables and plots.

That makes the correctness claim auditable instead of anecdotal.
