---
title: Remote Serving
description: Serve .kdb files or weighted mixtures over gRPC.
---

`kino-serve` exposes kinodb datasets over gRPC. It can serve a single `.kdb` file or a weighted mixture.

## Start A Server

Single file:

```bash
kino-serve data.kdb --port 50051
```

Weighted mixture:

```bash
kino-serve \
  --source bridge.kdb:0.4 \
  --source aloha.kdb:0.6 \
  --port 50051 \
  --seed 42
```

Bind address:

```bash
kino-serve data.kdb --bind 127.0.0.1 --port 50051
```

## Service Methods

The protobuf lives at `crates/kinodb-serve/proto/kinodb.proto`.

| RPC | Purpose |
| --- | --- |
| `ServerInfo` | Get episode count, frame count, source list, and server version |
| `GetMeta` | Fetch metadata for one episode |
| `GetEpisode` | Fetch a full episode |
| `GetBatch` | Fetch random, weighted, or sequential batches |
| `Query` | Run KQL and return matching positions plus metadata |

## Python Client

The high-level client lives in `kinodb.remote`.

```python
from kinodb.remote import KinoClient

client = KinoClient("localhost:50051")

print(client.server_info())

meta = client.get_meta(0)
episode = client.get_episode(0)
batch = client.get_batch(batch_size=16, mode="random")
hits = client.query("success = true", limit=10)

client.close()
```

## Remote Dataset

```python
from kinodb.remote import KinoRemoteDataset
from torch.utils.data import DataLoader

dataset = KinoRemoteDataset(
    "localhost:50051",
    batch_size=32,
    include_images=False,
)

loader = DataLoader(dataset, batch_size=None)

for episode in loader:
    actions = episode["action"]
    states = episode["state"]
    break
```

`batch_size=None` is intentional here because the remote dataset already yields samples from server-side batches.

## Batch Modes

| Mode | Behavior |
| --- | --- |
| `sequential` | Reads from `offset`, wrapping at the end |
| `random` | Samples uniformly for single-file mode |
| `weighted` | Samples through the mixture backend |

For single-file mode, `random` and `weighted` are effectively the same. For mixture mode, weighted sampling uses the source weights.

## Images

`GetBatch` has `include_images`. Keep it false unless the training worker genuinely needs image payloads over gRPC.

```python
batch = client.get_batch(8, include_images=True)
```

For large image datasets, local mmap reads or future shared-memory serving are better fits than shipping decoded frames over RPC.

## Current Limitations

- The Python gRPC stubs must be generated from `kinodb.proto`.
- Transport is insecure gRPC for local/trusted networks.
- Shared-memory serving was in the original blueprint but is not implemented in the current tree.
- The server reads complete episodes, not fixed-length windows.
