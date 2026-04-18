---
title: Installation
description: Build the kinodb CLI and Python bindings from source.
---

kinodb is currently a source-built project. The CLI is Rust, and the Python package is built separately through PyO3/maturin.

## Prerequisites

| Dependency | Why |
| --- | --- |
| Rust stable | Builds `kino`, `kino-serve`, and the core crates |
| CMake | Needed by native dependencies, especially HDF5 builds |
| HDF5 development libraries | Required by the `hdf5` Rust crate used for HDF5 ingest |
| Python 3.8+ | Required for the Python bindings |
| maturin | Builds and installs the PyO3 extension module |

## macOS

```bash
brew install cmake
brew install hdf5@1.10
export HDF5_DIR="$(brew --prefix hdf5@1.10)"
```

:::caution[Homebrew HDF5 version]
The local verification run on April 18, 2026 failed with Homebrew HDF5 `1.14.6` because `hdf5-sys 0.8.1` rejected that version string during its build script. Use `hdf5@1.10` for now, or pin another version known to work with `hdf5-sys 0.8.1`.
:::

## Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libhdf5-dev
```

## Build The CLI

From the repository root:

```bash
cargo build --release
target/release/kino --help
target/release/kino-serve --help
```

For convenience during local development:

```bash
export PATH="$PWD/target/release:$PATH"
kino --help
```

## Build Python Bindings

The Python package is intentionally excluded from the top-level Cargo workspace. Build it from its crate directory:

```bash
python -m pip install --upgrade pip
python -m pip install maturin numpy

cd crates/kinodb-py
maturin develop --release
cd ../..
```

Verify:

```bash
python - <<'PY'
import kinodb
print(kinodb.__version__)
print(kinodb.open)
PY
```

## Optional Python Dependencies

```bash
# PyTorch integration
python -m pip install torch torchvision

# gRPC client
python -m pip install grpcio grpcio-tools protobuf

# Benchmark and data conversion helpers
python -m pip install h5py pyarrow numpy matplotlib huggingface_hub
```

## Smoke Test

```bash
kino create-test /tmp/kinodb-smoke.kdb -n 10 --frames 32 --compress 85
kino info /tmp/kinodb-smoke.kdb
kino schema /tmp/kinodb-smoke.kdb
kino query /tmp/kinodb-smoke.kdb "num_frames >= 32"
```

```bash
python - <<'PY'
import kinodb
db = kinodb.open("/tmp/kinodb-smoke.kdb")
print(db.summary())
ep = db.read_episode(0)
print(ep["actions"].shape, ep["states"].shape)
PY
```

## Docs Site

The documentation is a separate Astro Starlight site:

```bash
cd kinodb-docs
npm install
npm run dev
```

Open `http://localhost:4321/kinodb/` when running with the GitHub Pages base path. For a root-local preview, remove or temporarily override `base: '/kinodb'` in `astro.config.mjs`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Invalid H5_VERSION` from `hdf5-sys` | Pin HDF5 to `hdf5@1.10` on macOS and export `HDF5_DIR` |
| `maturin develop` cannot import Rust crates | Run it from `crates/kinodb-py` |
| Python import finds an old extension | Re-run `maturin develop --release` in your active virtualenv |
| CLI feels slow | Make sure you are running `target/release/kino`, not a debug build |
| gRPC client cannot import stubs | Generate Python protobuf stubs from `crates/kinodb-serve/proto/kinodb.proto` |
