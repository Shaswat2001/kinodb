---
title: Architecture
description: Crates, data flow, and design decisions.
---

kinodb is a Rust workspace with a small Python package around the core reader.

```text
kinodb/
  Cargo.toml
  crates/
    kinodb-core/    # .kdb file format, reader, writer, KQL, mixtures
    kinodb-ingest/  # HDF5, LeRobot, RLDS ingest
    kinodb-cli/     # kino command
    kinodb-serve/   # kino-serve gRPC server
    kinodb-py/      # PyO3 bindings, built separately
```

## Data Flow

```text
HDF5 / LeRobot / RLDS
        │
        ▼
kinodb-ingest
        │
        ▼
KdbWriter ──► .kdb file ──► KdbReader
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
            kino CLI      Python API     kino-serve
```

## `kinodb-core`

Owns the file format and all source-independent behavior.

| Module | Responsibility |
| --- | --- |
| `types` | `EpisodeMeta`, `Frame`, `ImageObs`, `Episode` |
| `header` | 64-byte file header |
| `index` | fixed-size episode index entries |
| `writer` | streaming `.kdb` writer |
| `reader` | mmap-backed `.kdb` reader |
| `kql` | parser and evaluator |
| `mixture` | weighted multi-file sampling |
| `prefetch` | prefetch-related helpers |

Design decisions:

- Keep file layout explicit and little-endian.
- Store episode payloads contiguously.
- Put the index at the end so the writer can stream episodes.
- Use mmap for low open cost and OS-managed paging.
- Keep KQL metadata-only so filtering does not decode frames.

## `kinodb-ingest`

Normalizes external formats into `Episode` objects and writes through `KdbWriter`.

| Module | Source |
| --- | --- |
| `hdf5` | robomimic/LIBERO-style `data/demo_*` files |
| `lerobot` | LeRobot v2/v3 Parquet datasets |
| `rlds` | TFRecord files with RLDS-style features |

The ingesters make format-specific choices, such as concatenating HDF5 `obs/*` state keys in sorted order and grouping LeRobot rows by `episode_index`.

## `kinodb-cli`

Wraps core and ingest functionality in `kino`.

Implemented commands:

- `create-test`
- `ingest`
- `info`
- `schema`
- `validate`
- `query`
- `mix`
- `merge`
- `export`
- `bench`

## `kinodb-serve`

Provides a gRPC service through `tonic` and `prost`.

Backends:

- single `.kdb` file,
- weighted `Mixture`.

RPCs:

- `ServerInfo`
- `GetMeta`
- `GetEpisode`
- `GetBatch`
- `Query`

## `kinodb-py`

Provides the Python bridge:

- `kinodb.open()`
- `Database`
- NumPy-returning episode reads
- KQL queries
- PyTorch helpers in `kinodb.torch`
- gRPC client helpers in `kinodb.remote`

The Python package is excluded from the workspace root because it is built with `maturin`.

## Why Not Just HDF5?

HDF5 is excellent at hierarchical arrays. kinodb adds trajectory-specific behavior:

- one episode index across the whole dataset,
- uniform metadata filtering,
- cross-format ingest,
- dataset mixtures,
- CLI validation/schema tools,
- Python training integration over the same `.kdb` layout.

## Why Not Just Parquet?

Parquet is strong for columnar analytics. Robot training often asks for all data for one episode or sampled windows from episodes. kinodb stores an episode contiguously and keeps a direct byte index for that access pattern.

## Why Rust?

The hot path needs:

- memory safety around mmap and binary parsing,
- predictable performance,
- native CLI distribution,
- async gRPC serving,
- Python bindings without putting the engine in Python.

Rust gives the project a systems layer while still letting researchers train from Python.

## Current Technical Debt

The docs intentionally preserve the honest current state:

- HDF5 build compatibility needs pinning or a static build strategy.
- Python dataset handles need more ergonomic multi-worker behavior.
- Image reads currently decode into NumPy arrays; lazy/raw compressed image paths would improve image-heavy workflows.
- Shared-memory serving and hardware decode remain roadmap items from the original PDF blueprint.
