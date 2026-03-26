<p align="center">
  <h1 align="center">kinodb</h1>
  <p align="center"><em>from Greek <b>kínēsis</b> (motion) + <b>db</b> (database)</em></p>
  <p align="center">A high-performance trajectory database for robot learning. Built in Rust.</p>
</p>

<p align="center">
  <a href="#install">Install</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#why">Why?</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

**kinodb** ingests robot trajectory data from HDF5 (robomimic, LIBERO, DROID), stores it in a compact binary format (`.kdb`), and will serve training batches at GPU-saturating speeds.

```bash
# Ingest a robomimic/LIBERO HDF5 file
kino ingest data.hdf5 --output data.kdb --embodiment franka --task "pick up the block"

# Inspect the result
kino info data.kdb
```

```
kinodb v0.1

  File:      data.kdb
  Episodes:  200
  Frames:    18400
  Avg len:   92.0 frames/episode
  File size: 1.2 MB

  Embodiments (1):
    - franka

  Tasks (1):
    - pick up the block

  Success:   187/200 (93.5%)
```

> **Status:** 🚧 Early development — the core storage engine and HDF5 ingest work. LeRobot/RLDS ingest, batch serving, and Python bindings are coming.

## Install

### Prerequisites

- **Rust** (1.70+): https://rustup.rs
- **CMake**: needed to build the bundled HDF5 C library

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt install cmake
```

### Build from source

```bash
git clone https://github.com/YOUR_USERNAME/kinodb.git
cd kinodb
cargo build --release
```

The binary is at `target/release/kino`.

### Verify

```bash
cargo test
```

## Quick Start

### Generate test data

```bash
# Create a sample .kdb with fake robot episodes
kino create-test demo.kdb -n 20 --frames 50

# With fake camera images
kino create-test demo_images.kdb -n 10 --frames 30 --images
```

### Ingest real HDF5 data

```bash
# robomimic / LIBERO style HDF5 files
kino ingest path/to/dataset.hdf5 \
  --output dataset.kdb \
  --embodiment franka \
  --task "open the drawer" \
  --fps 20.0

# Only ingest the first 50 episodes
kino ingest large_dataset.hdf5 --output small.kdb --max-episodes 50
```

### Inspect a database

```bash
# Summary
kino info dataset.kdb

# Per-episode table
kino info --episodes dataset.kdb
```

## Why?

Every team training VLAs (Vision-Language-Action models) hits the same data bottleneck:

| Problem | Evidence |
|---------|----------|
| **Video decode is slower than backprop** | [LeRobot #1623](https://github.com/huggingface/lerobot/issues/1623): "Training SmolVLA spends more time waiting for the dataloader (1s) than running backprop (0.7s)" |
| **Datasets don't fit in memory** | [LeRobot #1346](https://github.com/huggingface/lerobot/issues/1346): GR00T 1.5 fine-tuning needs 937 GB for images alone |
| **Format fragmentation** | [LeRobot #2446](https://github.com/huggingface/lerobot/issues/2446): "Almost 600 variants of libero on the hub" |
| **RLDS is bloated** | [Robo-DM (ICRA 2025)](https://autolab.berkeley.edu): RLDS is 18–73× larger per episode than necessary |

**kinodb** solves this by being a purpose-built database engine:

- **Format-agnostic ingest**: reads HDF5, RLDS, LeRobot — you never convert between formats again
- **Compact storage**: binary columnar format with video-aware compression
- **O(1) random access**: episode index with byte offsets — jump to any frame instantly
- **Zero Python dependencies**: `cargo install kinodb` — no conda, no TensorFlow
- **No lock-in**: export back to standard formats anytime

## Architecture

### File format (`.kdb`)

```
┌──────────────────────────┐  byte 0
│  FileHeader (64 bytes)   │  magic "KINO", version, counts
├──────────────────────────┤  byte 64
│  Episode 0: meta blob    │  embodiment, task, fps, success
│  Episode 0: actions      │  state + action vectors (packed f32)
│  Episode 0: images       │  camera frames (raw RGB)
├──────────────────────────┤
│  Episode 1: ...          │
├──────────────────────────┤
│  ...                     │
├──────────────────────────┤  ← header.index_offset
│  Episode Index           │  N × 64-byte entries with byte offsets
└──────────────────────────┘
```

### Crate structure

```
kinodb/
├── crates/
│   ├── kinodb-core/       # Storage engine: types, header, index, reader, writer
│   ├── kinodb-ingest/     # Format readers (HDF5, soon: LeRobot, RLDS)
│   └── kinodb-cli/        # "kino" binary
├── Cargo.toml             # Workspace root
└── README.md
```

### Supported HDF5 structure

kinodb auto-discovers the robomimic/LIBERO convention:

```
data/
  demo_0/
    actions             (N, action_dim) float32    ← required
    rewards             (N,) float32               ← optional
    dones               (N,) float32               ← optional
    obs/
      agentview_image   (N, H, W, 3) uint8         ← auto-detected as camera
      robot0_eef_pos    (N, D) float32              ← auto-detected as state
      robot0_gripper_qpos (N, D) float32            ← auto-detected as state
  demo_1/
    ...
```

## CLI Reference

```
USAGE: kino <COMMAND>

  ingest        Import from HDF5 (robomimic/LIBERO) into .kdb
  info          Show database summary
  create-test   Generate sample .kdb with fake data
```

## Roadmap

- [x] Core storage engine (`.kdb` format, reader, writer)
- [x] HDF5 ingest (robomimic / LIBERO)
- [x] CLI (`kino info`, `kino ingest`, `kino create-test`)
- [ ] LeRobot v3 ingest
- [ ] RLDS ingest (TFRecord parser, no TF dependency)
- [ ] KQL query language (filter by embodiment, task, success)
- [ ] `kino mix` — weighted dataset mixtures
- [ ] gRPC batch serving
- [ ] Shared memory serving (zero-copy)
- [ ] PyTorch DataLoader (`pip install kinodb`)
- [ ] `kino export` — write back to HDF5/LeRobot/NumPy
- [ ] Video-aware image compression (H.264/H.265 segments)
- [ ] Hardware-accelerated decode (NVDEC/VAAPI)

## License

MIT OR Apache-2.0
