---
title: File Format
description: Binary layout of the current .kdb format.
---

The current `.kdb` format is a single little-endian file with a 64-byte header, contiguous episode payloads, and a fixed-size episode index at the end.

```text
byte 0
┌────────────────────────────┐
│ FileHeader                 │ 64 bytes
├────────────────────────────┤
│ Episode 0 metadata blob    │
│ Episode 0 state/action data│
│ Episode 0 image data       │
├────────────────────────────┤
│ Episode 1 metadata blob    │
│ Episode 1 state/action data│
│ Episode 1 image data       │
├────────────────────────────┤
│ ...                        │
├────────────────────────────┤
│ EpisodeIndex               │ N x 64 bytes
└────────────────────────────┘
```

The writer reserves the header first, writes all episodes, writes the index, then seeks back to fill the final header with counts and offsets.

## Header

`FileHeader` is exactly 64 bytes.

| Offset | Size | Field | Description |
| --- | ---: | --- | --- |
| 0 | 4 | `magic` | `KINO` |
| 4 | 2 | `version_major` | currently `0` |
| 6 | 2 | `version_minor` | currently `1` |
| 8 | 8 | `num_episodes` | episode count |
| 16 | 8 | `num_frames` | total frames |
| 24 | 8 | `index_offset` | byte offset to index |
| 32 | 8 | `index_length` | byte length of index |
| 40 | 8 | `created_timestamp` | Unix seconds |
| 48 | 16 | reserved | zeroed |

Readers reject:

- files shorter than 64 bytes,
- bad magic bytes,
- newer major versions.

Newer minor versions are accepted.

## Index Entry

Each `IndexEntry` is exactly 64 bytes.

| Offset | Size | Field | Description |
| --- | ---: | --- | --- |
| 0 | 8 | `episode_id` | assigned sequentially by writer |
| 8 | 4 | `num_frames` | frames in episode |
| 12 | 2 | `action_dim` | action vector dimension |
| 14 | 2 | `state_dim` | state vector dimension |
| 16 | 8 | `actions_offset` | byte offset to state/action section |
| 24 | 8 | `actions_length` | byte length of state/action section |
| 32 | 8 | `images_offset` | byte offset to image section |
| 40 | 8 | `images_length` | byte length of image section |
| 48 | 8 | `meta_offset` | byte offset to metadata blob |
| 56 | 8 | `meta_length` | byte length of metadata blob |

The index is stored at the end so appending episodes during write does not require knowing final offsets ahead of time.

## Metadata Blob

The current metadata encoding is deliberately simple:

```text
u16 embodiment_len
u8[embodiment_len] embodiment_utf8
u16 task_len
u8[task_len] task_utf8
f32 fps
u8 success        # 0 unknown, 1 false, 2 true
u8 reward_present # 0 absent, 1 present
f32 total_reward  # only if reward_present = 1
```

`task` is called `language_instruction` in the Rust type.

## State/Action Section

For each frame:

```text
f32[state_dim] state
f32[action_dim] action
f32 reward
u8 is_terminal
```

The writer checks every frame's action length against `action_dim`. Empty episodes are rejected.

## Image Section

For each frame:

```text
u16 num_cameras
for each camera:
  u16 camera_name_len
  u8[camera_name_len] camera_name_utf8
  u32 width
  u32 height
  u8 channels
  u8 format          # 0 raw, 1 compressed
  u32 data_len
  u8[data_len] data
```

When `format >= 1`, the reader attempts to decode JPEG/PNG bytes to raw RGB. If decode fails for a frame image, that image is skipped.

## Access Paths

| Operation | Reads |
| --- | --- |
| Open file | header and index through mmap |
| `read_meta(i)` | metadata blob only |
| `read_episode(i)` | metadata, state/action section, image section |
| `read_episode_actions_only(i)` | metadata and state/action section |

## Forward Compatibility

The current format is versioned as `0.1`. Compatibility rules are conservative:

- future major versions should be rejected by old readers,
- future minor versions may be accepted,
- index entries keep offsets and lengths, so metadata/image encodings can evolve behind those boundaries.
