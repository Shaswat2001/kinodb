---
title: Python API
description: Reference for kinodb.open, Database, PyTorch helpers, and the remote client.
---

## Module

```python
import kinodb
```

Exports:

| Name | Description |
| --- | --- |
| `kinodb.open(path)` | Open a `.kdb` file |
| `kinodb.Database` | Rust-backed reader class |
| `kinodb.__version__` | Python package version |

## `kinodb.open(path)`

```python
db = kinodb.open("data.kdb")
```

Returns a `Database`. Raises `IOError` if the file cannot be opened or parsed.

## `Database`

### Metadata

```python
len(db)
db.num_episodes()
db.num_frames()
db.path()
db.version()
db.summary()
```

| Method | Returns |
| --- | --- |
| `num_episodes()` | number of episodes |
| `num_frames()` | total frames |
| `path()` | source path |
| `version()` | file format version string |
| `summary()` | human-readable summary |
| `__len__()` | same as `num_episodes()` |

### `read_meta(position)`

```python
meta = db.read_meta(0)
```

Returns:

```python
{
    "episode_id": 0,
    "embodiment": "franka",
    "task": "open the drawer",
    "num_frames": 92,
    "fps": 20.0,
    "action_dim": 7,
    "success": True,
    "total_reward": 1.0,
}
```

### `read_episode(position)`

```python
episode = db.read_episode(0)
```

Returns:

| Key | Type |
| --- | --- |
| `meta` | dict |
| `actions` | NumPy `float32`, `(T, action_dim)` |
| `states` | NumPy `float32`, `(T, state_dim)` |
| `rewards` | NumPy `float32`, `(T,)` |
| `is_terminal` | list of bool |
| `images` | dict camera -> NumPy `uint8`, `(T, H, W, C)` |

### `read_episode_actions_only(position)`

```python
episode = db.read_episode_actions_only(0)
```

Same shape for metadata/actions/states/rewards, but skips image decode and omits `images`.

Use this for low-dimensional policies or training benchmarks where images are not part of the measured path.

### `query(query_str)`

```python
positions = db.query("success = true AND num_frames > 50")
```

Returns a list of matching episode positions.

Raises `ValueError` on parse errors.

## PyTorch Helpers

```python
from kinodb.torch import KinoDataset, KinoMapDataset, KinoIterDataset, from_mixture
```

### `KinoMapDataset`

```python
dataset = KinoDataset(
    "data.kdb",
    kql_filter="success = true",
    image_key="front",
    image_size=(224, 224),
    to_tensor=True,
)
```

Arguments:

| Argument | Type | Description |
| --- | --- | --- |
| `db_path` | str | `.kdb` path |
| `kql_filter` | str or None | Optional KQL filter |
| `image_key` | str or None | Camera name, first camera if omitted |
| `image_size` | tuple or None | Resize to `(H, W)` |
| `to_tensor` | bool | Convert NumPy arrays to Torch tensors |

### `KinoIterDataset`

```python
dataset = KinoIterDataset(
    "data.kdb",
    shuffle=True,
    seed=42,
)
```

Can also receive `{path: weight}` for multi-source iteration.

### `from_mixture`

```python
dataset = from_mixture(
    {"bridge.kdb": 0.4, "aloha.kdb": 0.6},
    seed=42,
)
```

Returns a `KinoIterDataset`.

## Remote Client

```python
from kinodb.remote import KinoClient, KinoRemoteDataset
```

### `KinoClient`

```python
client = KinoClient("localhost:50051")
info = client.server_info()
meta = client.get_meta(0)
episode = client.get_episode(0)
batch = client.get_batch(32, mode="random", include_images=False)
hits = client.query("success = true", limit=10)
client.close()
```

### `KinoRemoteDataset`

```python
dataset = KinoRemoteDataset(
    "localhost:50051",
    batch_size=32,
    include_images=False,
    to_tensor=True,
)
```

Yields episodes from server-side batches.

## Error Model

| Situation | Python error |
| --- | --- |
| Cannot open file | `IOError` |
| Invalid episode position | `IndexError` |
| KQL parse error | `ValueError` |
| NumPy shape conversion issue | `ValueError` |
| Missing gRPC dependency | `ImportError` |
