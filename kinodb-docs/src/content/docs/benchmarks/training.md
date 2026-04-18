---
title: Training Pipeline
description: End-to-end policy training curves, mixed-source training, and KQL interoperability results.
---

The training benchmarks answer the question that matters most for robot learning: when data loading is measured inside the training loop, does kinodb move wall-clock time?

The latest experiment run says yes. For CNN and MLP policies, kinodb turns the loader from the dominant cost into a smaller part of the step. For ViT policies, the win is still visible but smaller because model compute dominates.

:::note
These results come from the pasted run log at `/home/ubuntu/shaswat/kinodb/neurips_experiments/results/`. The raw JSON files are not committed in this repository yet, so this page is a launch report rather than a fully reproducible artifact.
:::

## Experiment 1: Training Curves

Setup:

- datasets: `pusht_image` and `libero_spatial`;
- policies: `cnn_bc`, `mlp`, and `vit`;
- seeds: `42`, `123`, and `456`;
- skipped: `robomimic_lift`;
- metric: native loader vs kinodb loader inside the measured training loop.

<div class="kino-result-grid">
  <div class="kino-result-card">
    <span>Best total speedup</span>
    <b>8.0x</b>
    <em>LIBERO spatial + MLP, seed 42</em>
  </div>
  <div class="kino-result-card">
    <span>Data time cut</span>
    <b>~9x</b>
    <em>LIBERO data step: ~34s native to ~3.6s kinodb</em>
  </div>
  <div class="kino-result-card">
    <span>Image data cut</span>
    <b>~8x</b>
    <em>PushT image data step: ~45s native to ~5.6s kinodb</em>
  </div>
  <div class="kino-result-card">
    <span>Compute-heavy floor</span>
    <b>2.2-2.4x</b>
    <em>ViT still improves, but compute dominates the step</em>
  </div>
</div>

### Total Speedup By Dataset And Policy

<div class="kino-bar-list" aria-label="End-to-end training speedups">
  <div class="kino-bar-row">
    <span>LIBERO spatial / MLP</span>
    <div class="kino-track"><i style="--w: 96%;"></i></div>
    <strong>7.7x +/- 0.2</strong>
  </div>
  <div class="kino-bar-row">
    <span>LIBERO spatial / CNN BC</span>
    <div class="kino-track"><i style="--w: 89%;"></i></div>
    <strong>7.1x +/- 0.0</strong>
  </div>
  <div class="kino-bar-row">
    <span>PushT image / CNN BC</span>
    <div class="kino-track"><i style="--w: 85%;"></i></div>
    <strong>6.8x +/- 0.0</strong>
  </div>
  <div class="kino-bar-row">
    <span>PushT image / MLP</span>
    <div class="kino-track"><i style="--w: 83%;"></i></div>
    <strong>6.6x +/- 0.0</strong>
  </div>
  <div class="kino-bar-row">
    <span>PushT image / ViT</span>
    <div class="kino-track"><i style="--w: 30%;"></i></div>
    <strong>2.4x +/- 0.0</strong>
  </div>
  <div class="kino-bar-row">
    <span>LIBERO spatial / ViT</span>
    <div class="kino-track"><i style="--w: 28%;"></i></div>
    <strong>2.2x +/- 0.0</strong>
  </div>
</div>

| Dataset | Policy | Init speedup | Total speedup |
| --- | --- | ---: | ---: |
| `libero_spatial` | `cnn_bc` | 1394x | 7.1x +/- 0.0 |
| `libero_spatial` | `mlp` | 1364x | 7.7x +/- 0.2 |
| `libero_spatial` | `vit` | 2828x | 2.2x +/- 0.0 |
| `pusht_image` | `cnn_bc` | 26x | 6.8x +/- 0.0 |
| `pusht_image` | `mlp` | 31x | 6.6x +/- 0.0 |
| `pusht_image` | `vit` | 31x | 2.4x +/- 0.0 |

### What The Curve Logs Show

The losses match between native and kinodb for the same dataset, policy, seed, and epoch. The win is wall-clock time, not a different learning target.

| Run | Native data per epoch | kinodb data per epoch | Native compute | kinodb compute | Total speedup | Data share shift |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PushT image + CNN BC | ~44-45s | ~5.4-5.5s | ~1.7-1.9s | ~1.2-1.5s | 6.7-6.8x | 96% native to 79-80% kinodb |
| PushT image + MLP | ~44-45s | ~5.5-5.7s | ~1.5-1.7s | ~1.2-1.4s | 6.6-6.7x | 96% native to 80-81% kinodb |
| PushT image + ViT | ~45s | ~5.8s | ~23s | ~22.6s | 2.4x | 66% native to 20% kinodb |
| LIBERO spatial + CNN BC | ~34s | ~3.6-3.7s | ~1.7-1.9s | ~1.3-1.5s | 7.1x | 93% native to 71% kinodb |
| LIBERO spatial + MLP | ~34s | ~3.6-3.8s | ~1.2-1.5s | ~0.9-1.2s | 7.6-8.0x | 89-94% native to 76% kinodb |
| LIBERO spatial + ViT | ~34-35s | ~3.8-3.9s | ~22.6-24.6s | ~23s | 2.2-2.3x | 57-59% native to 14% kinodb |

Representative matching losses:

| Dataset / policy / seed | Native epoch 20 | kinodb epoch 20 |
| --- | ---: | ---: |
| PushT image / CNN BC / 42 | 1445.6413 | 1445.6413 |
| PushT image / MLP / 123 | 1294.5226 | 1294.5226 |
| LIBERO spatial / CNN BC / 42 | 0.0481 | 0.0481 |
| LIBERO spatial / ViT / 456 | 0.0403 | 0.0403 |

## Experiment 2: Interoperability

The interoperability experiment tests the main systems claim: after conversion, training code can sample from multiple robotics formats through one kinodb path.

<div class="kino-result-grid">
  <div class="kino-result-card">
    <span>Loader code</span>
    <b>8 LOC</b>
    <em>kinodb mixed-source loader vs 26 native LOC</em>
  </div>
  <div class="kino-result-card">
    <span>Sources mixed</span>
    <b>4</b>
    <em>PushT image, LIBERO image, PushT state, ALOHA insertion</em>
  </div>
  <div class="kino-result-card">
    <span>Schema padding</span>
    <b>14 / 14</b>
    <em>max state and action dimensions across sources</em>
  </div>
  <div class="kino-result-card">
    <span>Final logged loss</span>
    <b>33.6219</b>
    <em>after 15 logged epochs on the mixed dataset</em>
  </div>
</div>

Mixed sources:

| Source | Weight |
| --- | ---: |
| `lerobot_pusht_image.kdb` | 0.25 |
| `lerobot_libero_spatial_image.kdb` | 0.25 |
| `lerobot_pusht.kdb` | 0.25 |
| `lerobot_aloha_sim_insertion_scripted.kdb` | 0.25 |

Mixed-source training log:

| Epoch | Loss | Time |
| ---: | ---: | ---: |
| 0 | 5559.1095 | 0.642s |
| 5 | 360.9029 | 0.179s |
| 10 | 57.7633 | 0.182s |
| 15 | 33.6219 | 0.179s |

### KQL Query Latency

The KQL scan stays in microseconds on these converted datasets.

| Dataset | Query | Matches | Time |
| --- | --- | ---: | ---: |
| ALOHA insertion | `num_frames > 50` | 50/50 | 1620us |
| ALOHA insertion | `num_frames > 100` | 50/50 | 10us |
| LIBERO spatial image | `num_frames > 50` | 432/432 | 2890us |
| LIBERO spatial image | `num_frames > 100` | 354/432 | 106us |
| PushT state | `num_frames > 50` | 205/206 | 97us |
| PushT state | `num_frames > 100` | 152/206 | 33us |
| PushT image | `num_frames > 50` | 206/206 | 1205us |
| PushT image | `num_frames > 100` | 201/206 | 33us |

## Interpretation

kinodb helps most when the native loader is the bottleneck. That is exactly what the data-share percentages show: CNN/MLP runs spend roughly 89-96% of native time in data loading, then drop to 71-81% with kinodb. ViT runs are still faster, but their compute budget is so large that loader gains have a smaller ceiling.

The mixed-source experiment is equally important: the result is not only faster loading, but fewer custom loaders and one training path across datasets with different state/action dimensions.
