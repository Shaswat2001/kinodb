---
title: IO Performance
description: Scaling, storage, and image-throughput benchmark results from the latest experiment run.
---

This page presents the systems benchmarks from the latest pasted experiment log. It focuses on open time, sequential reads, KQL latency, storage size, write speed, and image-throughput validation.

## Scaling

Synthetic datasets were generated at 100, 500, 1K, 5K, 10K, and 50K episodes, with 50 frames per episode. Each run generated HDF5, Parquet, ingested to kinodb, then measured open, sequential read, and KQL.

<div class="kino-result-grid">
  <div class="kino-result-card">
    <span>50K open time</span>
    <b>1.2ms</b>
    <em>kinodb vs 158.1ms HDF5 and 2.87s Parquet</em>
  </div>
  <div class="kino-result-card">
    <span>50K sequential read</span>
    <b>1.26s</b>
    <em>kinodb vs 13.36s HDF5 and 118.40s Parquet</em>
  </div>
  <div class="kino-result-card">
    <span>50K KQL scan</span>
    <b>31.7ms</b>
    <em>metadata query across 50K episodes</em>
  </div>
  <div class="kino-result-card">
    <span>Parquet seq gap</span>
    <b>94x</b>
    <em>at 50K episodes: 118.40s vs 1.26s</em>
  </div>
</div>

### 50K Episode Snapshot

<div class="kino-compare-bars" aria-label="50K episode benchmark snapshot">
  <div class="kino-compare-row">
    <span>Open / HDF5</span>
    <div class="kino-track"><i style="--w: 6%;"></i></div>
    <strong>158.1ms</strong>
  </div>
  <div class="kino-compare-row">
    <span>Open / Parquet</span>
    <div class="kino-track"><i style="--w: 100%;"></i></div>
    <strong>2.87s</strong>
  </div>
  <div class="kino-compare-row is-kdb">
    <span>Open / kinodb</span>
    <div class="kino-track"><i style="--w: 1%;"></i></div>
    <strong>1.2ms</strong>
  </div>
  <div class="kino-compare-row">
    <span>Sequential / HDF5</span>
    <div class="kino-track"><i style="--w: 11%;"></i></div>
    <strong>13.36s</strong>
  </div>
  <div class="kino-compare-row">
    <span>Sequential / Parquet</span>
    <div class="kino-track"><i style="--w: 100%;"></i></div>
    <strong>118.40s</strong>
  </div>
  <div class="kino-compare-row is-kdb">
    <span>Sequential / kinodb</span>
    <div class="kino-track"><i style="--w: 1%;"></i></div>
    <strong>1.26s</strong>
  </div>
</div>

### Full Scaling Table

| Episodes | HDF5 open | Parquet open | kinodb open | HDF5 seq | Parquet seq | kinodb seq | KQL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 912us | 18.1ms | 51us | 26.0ms | 50.1ms | 2.2ms | 88us |
| 500 | 1.6ms | 26.9ms | 133us | 127.5ms | 226.7ms | 10.7ms | 259us |
| 1,000 | 2.4ms | 48.9ms | 82us | 251.8ms | 480.2ms | 18.8ms | 507us |
| 5,000 | 10.0ms | 312.8ms | 158us | 1.30s | 3.22s | 90.1ms | 2.9ms |
| 10,000 | 19.0ms | 666.8ms | 262us | 2.61s | 8.30s | 181.4ms | 5.7ms |
| 50,000 | 158.1ms | 2.87s | 1.2ms | 13.36s | 118.40s | 1.26s | 31.7ms |

The scaling shape is the result: kinodb keeps open and metadata access near-index-bound, while Parquet open/read costs grow sharply with many small trajectory groups.

## Storage Efficiency

The storage experiment tested state-only data and image-heavy data across HDF5, compressed HDF5, NPY directory layouts, Parquet, and kinodb.

### State-Only Storage

| Dataset size | HDF5 | HDF5 compressed | NPY dir | Parquet | kinodb |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 eps x 50 frames | 0.64 MB | 1.40 MB | 0.48 MB | 0.91 MB | **0.45 MB** |
| 500 eps x 50 frames | 3.19 MB | 6.98 MB | 2.39 MB | 4.02 MB | **2.26 MB** |
| 1,000 eps x 50 frames | 6.37 MB | 13.96 MB | 4.78 MB | 7.45 MB | **4.52 MB** |

Write time for the 1,000-episode state-only case:

<div class="kino-bar-list" aria-label="State-only write time">
  <div class="kino-bar-row">
    <span>HDF5</span>
    <div class="kino-track"><i style="--w: 43%;"></i></div>
    <strong>0.542s</strong>
  </div>
  <div class="kino-bar-row">
    <span>HDF5 compressed</span>
    <div class="kino-track"><i style="--w: 100%;"></i></div>
    <strong>1.250s</strong>
  </div>
  <div class="kino-bar-row">
    <span>NPY dir</span>
    <div class="kino-track"><i style="--w: 20%;"></i></div>
    <strong>0.247s</strong>
  </div>
  <div class="kino-bar-row">
    <span>Parquet</span>
    <div class="kino-track"><i style="--w: 32%;"></i></div>
    <strong>0.400s</strong>
  </div>
  <div class="kino-bar-row is-kdb">
    <span>kinodb</span>
    <div class="kino-track"><i style="--w: 2%;"></i></div>
    <strong>0.018s</strong>
  </div>
</div>

### Image Storage

For image-heavy synthetic data, kinodb lands at storage parity with raw layouts and writes faster than Parquet. HDF5 compression is not helpful on these synthetic images; it increases size slightly and makes writes much slower.

| Dataset size | HDF5 | HDF5 compressed | NPY dir | Parquet | kinodb |
| --- | ---: | ---: | ---: | ---: | ---: |
| 84x84, 100 eps x 30 frames | 64.10 MB | 72.08 MB | 63.82 MB | 64.06 MB | **63.78 MB** |
| 84x84, 500 eps x 30 frames | 320.48 MB | 360.38 MB | 319.10 MB | 320.17 MB | **318.90 MB** |
| 224x224, 50 eps x 30 frames | 226.09 MB | 227.73 MB | 225.95 MB | 226.08 MB | **225.93 MB** |
| 224x224, 200 eps x 30 frames | 904.35 MB | 910.91 MB | 903.80 MB | 904.32 MB | **903.72 MB** |

Representative write times:

| Case | HDF5 | HDF5 compressed | NPY dir | Parquet | kinodb |
| --- | ---: | ---: | ---: | ---: | ---: |
| 84x84, 500 episodes | 0.680s | 13.014s | 0.410s | 1.364s | **0.269s** |
| 224x224, 200 episodes | 0.835s | 25.685s | 0.705s | 4.867s | **0.808s** |

## Summary

The strongest systems claims from this run are:

- kinodb opens 50K-episode synthetic datasets in 1.2ms;
- kinodb sequentially reads the same 50K run in 1.26s vs 13.36s for HDF5 and 118.40s for Parquet;
- KQL metadata queries stay below 32ms at 50K episodes;
- state-only `.kdb` storage is the smallest format tested in the run;
- image-heavy `.kdb` storage is at native-size parity, with faster writes than Parquet in the reported cases.
