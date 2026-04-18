---
title: KQL Queries
description: Filter episodes with Kino Query Language.
---

KQL is the small metadata query language built into `kinodb-core`. It is intentionally narrow: filter episodes by fields needed for robot dataset selection, curriculum construction, validation, and mixed-source training.

## Syntax

```text
<field> <operator> <value> [AND <field> <operator> <value> ...]
```

Examples:

```bash
kino query data.kdb "success = true"
kino query data.kdb "embodiment = 'franka' AND num_frames > 50"
kino query data.kdb "task CONTAINS 'pick' AND fps >= 10.0"
kino query data.kdb "total_reward != null" --limit 25
```

## Fields

| Field | Type | Example |
| --- | --- | --- |
| `embodiment` | string | `embodiment = 'franka'` |
| `task` | string | `task CONTAINS 'drawer'` |
| `success` | bool or null | `success = true` |
| `num_frames` | int | `num_frames >= 100` |
| `action_dim` | int | `action_dim = 7` |
| `fps` | float | `fps >= 10.0` |
| `total_reward` | float or null | `total_reward > 0.5` |

`task` maps to `EpisodeMeta.language_instruction` in Rust and to `meta["task"]` in Python.

## Operators

| Operator | Meaning | Applies to |
| --- | --- | --- |
| `=` | equals | all fields |
| `!=` | not equals | all fields |
| `>` | greater than | numeric fields |
| `<` | less than | numeric fields |
| `>=` | greater than or equal | numeric fields |
| `<=` | less than or equal | numeric fields |
| `CONTAINS` | substring match | string fields |

KQL currently supports `AND`. `OR`, parentheses, projections, and joins are intentionally out of scope for the current implementation.

## Values

```text
'single quoted string'
"double quoted string"
bare_string
true
false
null
none
123
12.5
```

Bare words are accepted as strings:

```bash
kino query data.kdb "embodiment = franka"
```

For launch docs and scripts, quote strings anyway. It makes examples easier to read.

## CLI

```bash
kino query data.kdb "success = true AND task CONTAINS 'pick'"
kino query data.kdb "num_frames > 100" --limit 10
```

The command prints matching episode positions and metadata. Positions are zero-based and can be passed to Python `read_episode(position)`.

## Python

```python
import kinodb

db = kinodb.open("data.kdb")
positions = db.query("success = true AND num_frames > 100")

for pos in positions[:5]:
    meta = db.read_meta(pos)
    print(pos, meta["task"], meta["num_frames"])
```

## Training Filters

KQL filters can be used directly in `KinoDataset`:

```python
from kinodb.torch import KinoDataset

dataset = KinoDataset(
    "data.kdb",
    kql_filter="success = true AND action_dim = 7",
)
```

## Merge Filters

Create a smaller physical dataset:

```bash
kino merge raw.kdb --output successful.kdb --filter "success = true"
```

This is useful when distributing a curated dataset split.

## Performance Model

KQL works by parsing the expression into a small AST and scanning episode metadata with `read_meta`. It avoids decoding frames and images. That is why benchmark metadata scans are the strongest win: native HDF5/Parquet/RLDS loaders usually have to walk their own source structures, while `.kdb` keeps episode metadata addressable through the index.

Recorded benchmark summary:

| Dataset class | Metadata scan result |
| --- | --- |
| 10 tabular datasets | Median 375x faster, range 48-612x |
| 5 image datasets | 605-2,648x faster |

## Parser Errors

Common errors:

| Error | Cause |
| --- | --- |
| `empty query` | The string is empty or whitespace |
| `unknown field` | Field is not one of the supported KQL fields |
| `expected operator` | Missing `=`, `!=`, comparison, or `CONTAINS` |
| `unterminated string` | Missing closing quote |
| `expected AND` | KQL currently only supports `AND` between conditions |
