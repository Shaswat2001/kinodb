---
title: Training Pipeline
description: Data loading, batch construction, training-step, and inference benchmark notes.
---

The training benchmarks were designed to answer one question: does the storage speedup survive once data flows through a model loop?

The key lesson from the chat history is that the benchmark must include disk access. An earlier script loaded both native and `.kdb` data into NumPy arrays first, then timed PyTorch. That measured model compute after the data source had disappeared. The corrected benchmark reads from disk inside the measured path.

## Correct Methodology

Measure:

| Metric | What it includes |
| --- | --- |
| Load | `get_episode()` / native episode load from disk |
| Batch | load plus tensor/batch construction |
| Training step | load plus forward, backward, and optimizer step |
| Inference | load plus forward pass |

Avoid:

- preloading all episodes into RAM before timing,
- comparing native in-memory arrays to `.kdb` disk reads,
- using different state dimensions across native and `.kdb` paths,
- batching mixed action dimensions without padding or schema-aware collate logic.

## Recorded Results

The chat history records these corrected training/inference summaries:

| Dataset | Load | Batch | Train | Infer |
| --- | ---: | ---: | ---: | ---: |
| taco_play | 590x | 736x | 185x | 261x |
| xarm_lift | 125x | 117x | 19x | 31x |
| aloha_insertion | 35x | 50x | 35x | 35x |
| aloha_transfer | 44x | 53x | 44x | 37x |
| pusht | 38x | 21x | 15x | 12x |
| pusht_image | 5-20x | 16x | 12x | 12x |
| LIBERO image | 0.2-3x | 3.6-4.2x | 3.7-4.2x | 3.8-4.2x |

Policy-level launch snippets:

| Dataset x policy | Speedup |
| --- | ---: |
| PushT x MLP | about 9.4x |
| PushT x Diffusion-style policy | about 5.2x |
| PushT x ACT-style policy | about 1.6x |
| PushT x Transformer BC | about 4.5x |
| PushT image x CNN BC | about 7.8x |
| PushT image x ViT | about 4.7x |
| ALOHA insertion x MLP | about 5.8x |
| ALOHA insertion x ACT-style policy | about 1.5x |

The speedup shrinks as the model becomes compute-heavy. That is expected: when forward/backward dominates wall-clock time, improving data loading changes a smaller part of the total step.

## Bugs Found And Fixed

### Preload bug

The first training script loaded data into memory and then timed a `TensorDataset`. That made native and kinodb look artificially similar.

Fix: streaming mode where `__getitem__` performs the native read or `.kdb` read during the measured path.

### robomimic state mismatch

Native HDF5 benchmark code selected one observation key, while kinodb concatenated all 2D `obs/*` state keys. That caused dimension mismatches such as a native model using one state dimension and a `.kdb` model using the full concatenated state.

Fixes:

- native HDF5 state extraction returns all state keys,
- `_extract_state()` concatenates those keys,
- inference builds separate native and `.kdb` models if dimensions still differ.

### Mixed action dimensions

Mixing PushT and ALOHA failed because `torch.stack` cannot stack action vectors of size 2 and 14.

Fix: detect max state/action dimensions across sources and zero-pad smaller vectors, or use a schema-aware collate strategy.

## NeurIPS-Ready Experiments Still Needed

The benchmark numbers are good launch material. For a paper, the chat history correctly identified that they are not enough by themselves.

Priority experiments:

| Experiment | Why it matters |
| --- | --- |
| Real policy curves | Show loss vs wall-clock time for native vs kinodb, not only IO microbenchmarks |
| Interoperability demo | Train one model from HDF5 + LeRobot + RLDS converted into one mixture |
| Scaling plot | Show open/query behavior as episode count reaches 100K+ |
| Multi-worker scaling | Show behavior with `num_workers` across datasets |
| Ablation | mmap, index, actions-only read, JPEG pass-through |

The strongest paper figure would be a real Diffusion Policy or ACT training run where the learning curve matches native data but reaches the same loss earlier in wall-clock time.

## Reporting Guidance

Use careful language:

- "kinodb accelerates the data-loading portion of training" is precise.
- "kinodb makes all training 10x faster" is not true for compute-heavy policies.
- "image-heavy workloads need lazy/raw image paths" is honest and useful.
- "the central contribution is a unified trajectory data layer" is stronger than "a faster file format."
