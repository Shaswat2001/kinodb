#!/usr/bin/env python3
"""
kinodb vs HDF5 vs Parquet — Ablation Benchmark
================================================

Creates identical trajectory data in three formats, then measures:
  1. File size on disk
  2. Write time
  3. Sequential read (all episodes)
  4. Random access (100 random episodes)
  5. Filtered read (equivalent of KQL query)

Requirements:
  pip install h5py pyarrow numpy
  cd crates/kinodb-py && maturin develop --release

Usage:
  python benchmark_comparison.py
  python benchmark_comparison.py --episodes 500 --frames 100
  python benchmark_comparison.py --episodes 1000 --frames 50 --images
"""

import argparse
import json
import os
import random
import shutil
import struct
import tempfile
import time

import numpy as np

# ── Config ───────────────────────────────────────────────────

EMBODIMENTS = ["franka", "widowx", "aloha", "ur5"]
TASKS = [
    "pick up the red block",
    "open the drawer",
    "place cup on plate",
    "push the green button",
]

# ── Data generation ──────────────────────────────────────────

def generate_episodes(n_episodes, n_frames, action_dim=7, state_dim=6, with_images=False):
    """Generate synthetic episode data (shared across all formats)."""
    episodes = []
    for i in range(n_episodes):
        emb = EMBODIMENTS[i % len(EMBODIMENTS)]
        task = TASKS[i % len(TASKS)]
        success = i % 3 != 0

        actions = np.random.randn(n_frames, action_dim).astype(np.float32) * 0.01
        states = np.random.randn(n_frames, state_dim).astype(np.float32) * 0.1
        rewards = np.zeros(n_frames, dtype=np.float32)
        if success:
            rewards[-1] = 1.0

        images = None
        if with_images:
            images = np.random.randint(0, 255, (n_frames, 64, 64, 3), dtype=np.uint8)

        episodes.append({
            "embodiment": emb,
            "task": task,
            "success": success,
            "actions": actions,
            "states": states,
            "rewards": rewards,
            "images": images,
        })
    return episodes


# ── HDF5 format ──────────────────────────────────────────────

def write_hdf5(episodes, path):
    """Write episodes in robomimic-style HDF5."""
    import h5py
    t0 = time.perf_counter()
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, ep in enumerate(episodes):
            demo = data.create_group(f"demo_{i}")
            demo.create_dataset("actions", data=ep["actions"])
            demo.create_dataset("rewards", data=ep["rewards"])
            dones = np.zeros(len(ep["rewards"]), dtype=np.float32)
            dones[-1] = 1.0
            demo.create_dataset("dones", data=dones)
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=ep["states"][:, :3])
            obs.create_dataset("robot0_gripper_qpos", data=ep["states"][:, 3:])
            if ep["images"] is not None:
                obs.create_dataset("agentview_image", data=ep["images"])
    return time.perf_counter() - t0


def read_hdf5_sequential(path):
    """Read all episodes sequentially from HDF5."""
    import h5py
    t0 = time.perf_counter()
    total_frames = 0
    with h5py.File(path, "r") as f:
        data = f["data"]
        for key in sorted(data.keys()):
            demo = data[key]
            actions = demo["actions"][:]
            rewards = demo["rewards"][:]
            obs = demo["obs"]
            state_parts = []
            for obs_key in sorted(obs.keys()):
                ds = obs[obs_key]
                if len(ds.shape) == 2:  # state, not image
                    state_parts.append(ds[:])
            if state_parts:
                states = np.concatenate(state_parts, axis=1)
            total_frames += len(actions)
    dur = time.perf_counter() - t0
    return dur, total_frames


def read_hdf5_random(path, indices):
    """Read specific episodes by index from HDF5."""
    import h5py
    t0 = time.perf_counter()
    total_frames = 0
    with h5py.File(path, "r") as f:
        data = f["data"]
        keys = sorted(data.keys())
        for idx in indices:
            demo = data[keys[idx]]
            actions = demo["actions"][:]
            rewards = demo["rewards"][:]
            total_frames += len(actions)
    dur = time.perf_counter() - t0
    return dur, total_frames


def query_hdf5(path, embodiment="franka"):
    """Filter episodes by embodiment (no native query — must scan all metadata)."""
    import h5py
    t0 = time.perf_counter()
    # HDF5 has no metadata index, so we'd need to store embodiment as an attribute
    # or scan all episodes. In practice, people just load everything.
    # We simulate: read actions from every 4th episode (since embodiment cycles by 4)
    hits = 0
    total_frames = 0
    with h5py.File(path, "r") as f:
        data = f["data"]
        keys = sorted(data.keys())
        for i, key in enumerate(keys):
            # Simulate checking embodiment — in real HDF5 there's no metadata field for this
            if EMBODIMENTS[i % len(EMBODIMENTS)] == embodiment:
                demo = data[key]
                actions = demo["actions"][:]
                total_frames += len(actions)
                hits += 1
    dur = time.perf_counter() - t0
    return dur, hits, total_frames


# ── Parquet format (LeRobot-style) ───────────────────────────

def write_parquet(episodes, path):
    """Write episodes as a single Parquet file (LeRobot v3 style)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.perf_counter()

    rows = []
    for i, ep in enumerate(episodes):
        n = len(ep["actions"])
        for t in range(n):
            row = {
                "episode_index": i,
                "index": t,
                "timestamp": t / 10.0,
                "task_index": i % len(TASKS),
            }
            # Actions as individual columns (LeRobot style)
            for d in range(ep["actions"].shape[1]):
                row[f"action_{d}"] = float(ep["actions"][t, d])
            # State as individual columns
            for d in range(ep["states"].shape[1]):
                row[f"observation.state_{d}"] = float(ep["states"][t, d])
            rows.append(row)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="snappy")

    dur = time.perf_counter() - t0
    return dur


def read_parquet_sequential(path, n_episodes, action_dim=7, state_dim=6):
    """Read all episodes from Parquet."""
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    table = pq.read_table(path)
    df = table.to_pandas()

    total_frames = 0
    for ep_idx in range(n_episodes):
        ep_df = df[df["episode_index"] == ep_idx]
        action_cols = [f"action_{d}" for d in range(action_dim)]
        state_cols = [f"observation.state_{d}" for d in range(state_dim)]
        actions = ep_df[action_cols].values.astype(np.float32)
        states = ep_df[state_cols].values.astype(np.float32)
        total_frames += len(actions)

    dur = time.perf_counter() - t0
    return dur, total_frames


def read_parquet_random(path, indices, action_dim=7, state_dim=6):
    """Read specific episodes from Parquet (requires full scan + filter)."""
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    table = pq.read_table(path)
    df = table.to_pandas()

    total_frames = 0
    action_cols = [f"action_{d}" for d in range(action_dim)]
    for idx in indices:
        ep_df = df[df["episode_index"] == idx]
        actions = ep_df[action_cols].values.astype(np.float32)
        total_frames += len(actions)

    dur = time.perf_counter() - t0
    return dur, total_frames


def query_parquet(path, embodiment="franka", action_dim=7):
    """Filter Parquet by embodiment (must do external join since Parquet has no metadata)."""
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    table = pq.read_table(path)
    df = table.to_pandas()

    # Parquet doesn't store embodiment natively — we reconstruct it
    hits = 0
    total_frames = 0
    action_cols = [f"action_{d}" for d in range(action_dim)]
    n_episodes = df["episode_index"].max() + 1
    for ep_idx in range(n_episodes):
        if EMBODIMENTS[ep_idx % len(EMBODIMENTS)] == embodiment:
            ep_df = df[df["episode_index"] == ep_idx]
            actions = ep_df[action_cols].values.astype(np.float32)
            total_frames += len(actions)
            hits += 1

    dur = time.perf_counter() - t0
    return dur, hits, total_frames


# ── kinodb format ────────────────────────────────────────────

def write_kinodb(episodes, kdb_path):
    """Write episodes using kino CLI (create-test or ingest from HDF5)."""
    # We write via HDF5 intermediate since kinodb CLI reads HDF5
    tmp_hdf5 = kdb_path + ".tmp.hdf5"
    write_hdf5(episodes, tmp_hdf5)

    t0 = time.perf_counter()
    os.system(
        f"kino ingest {tmp_hdf5} --output {kdb_path} "
        f"--embodiment mixed --task mixed --fps 10 2>/dev/null"
    )
    dur = time.perf_counter() - t0

    os.remove(tmp_hdf5)
    return dur


def write_kinodb_direct(episodes, kdb_path):
    """Write episodes directly using the Python bindings + KdbWriter."""
    # If kinodb Python bindings support writing, use them.
    # Otherwise fall back to CLI.
    # For now, use the CLI approach via HDF5 intermediate.
    return write_kinodb(episodes, kdb_path)


def read_kinodb_sequential(kdb_path):
    """Read all episodes from .kdb using Python bindings."""
    import kinodb

    t0 = time.perf_counter()
    db = kinodb.open(kdb_path)
    total_frames = 0
    for i in range(db.num_episodes()):
        ep = db.read_episode(i)
        actions = np.array(ep["actions"], dtype=np.float32)
        states = np.array(ep["states"], dtype=np.float32)
        total_frames += len(actions)

    dur = time.perf_counter() - t0
    return dur, total_frames


def read_kinodb_random(kdb_path, indices):
    """Read specific episodes from .kdb."""
    import kinodb

    t0 = time.perf_counter()
    db = kinodb.open(kdb_path)
    total_frames = 0
    for idx in indices:
        ep = db.read_episode(idx)
        actions = np.array(ep["actions"], dtype=np.float32)
        total_frames += len(actions)

    dur = time.perf_counter() - t0
    return dur, total_frames


def query_kinodb(kdb_path, embodiment="franka"):
    """Filter episodes using KQL."""
    import kinodb

    t0 = time.perf_counter()
    db = kinodb.open(kdb_path)
    hits = db.query(f"embodiment = '{embodiment}'")

    total_frames = 0
    for idx in hits:
        ep = db.read_episode(idx)
        actions = np.array(ep["actions"], dtype=np.float32)
        total_frames += len(actions)

    dur = time.perf_counter() - t0
    return dur, len(hits), total_frames


# ── Main ─────────────────────────────────────────────────────

def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds*1_000_000:.0f} µs"
    elif seconds < 1.0:
        return f"{seconds*1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def format_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    else:
        return f"{b/1024**3:.2f} GB"


def main():
    parser = argparse.ArgumentParser(description="kinodb vs HDF5 vs Parquet benchmark")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--frames", type=int, default=50, help="Frames per episode")
    parser.add_argument("--images", action="store_true", help="Include 64x64 images")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    args = parser.parse_args()

    n_ep = args.episodes
    n_fr = args.frames
    n_random = min(100, n_ep)
    random_indices = random.sample(range(n_ep), n_random)

    print("=" * 72)
    print("kinodb vs HDF5 vs Parquet — Ablation Benchmark")
    print("=" * 72)
    print(f"  Episodes:       {n_ep}")
    print(f"  Frames/episode: {n_fr}")
    print(f"  Total frames:   {n_ep * n_fr}")
    print(f"  Action dim:     7")
    print(f"  State dim:      6")
    print(f"  Images:         {'64x64 RGB' if args.images else 'none'}")
    print(f"  Random sample:  {n_random} episodes")
    print(f"  Runs:           {args.runs}")
    print()

    # Generate data once
    print("Generating synthetic data...")
    episodes = generate_episodes(n_ep, n_fr, with_images=args.images)
    print()

    tmpdir = tempfile.mkdtemp(prefix="kinodb_bench_")
    hdf5_path = os.path.join(tmpdir, "data.hdf5")
    parquet_path = os.path.join(tmpdir, "data.parquet")
    kdb_path = os.path.join(tmpdir, "data.kdb")

    results = {}

    # ── Write ────────────────────────────────────────────────
    print("Writing...")
    hdf5_write = write_hdf5(episodes, hdf5_path)
    parquet_write = write_parquet(episodes, parquet_path)
    kdb_write = write_kinodb(episodes, kdb_path)

    hdf5_size = os.path.getsize(hdf5_path)
    parquet_size = os.path.getsize(parquet_path)
    kdb_size = os.path.getsize(kdb_path)

    results["write_time"] = {"HDF5": hdf5_write, "Parquet": parquet_write, "kinodb": kdb_write}
    results["file_size"] = {"HDF5": hdf5_size, "Parquet": parquet_size, "kinodb": kdb_size}

    # ── Sequential read (averaged) ───────────────────────────
    print("Sequential read...")
    seq_times = {"HDF5": [], "Parquet": [], "kinodb": []}
    for _ in range(args.runs):
        d, _ = read_hdf5_sequential(hdf5_path)
        seq_times["HDF5"].append(d)
        d, _ = read_parquet_sequential(parquet_path, n_ep)
        seq_times["Parquet"].append(d)
        d, _ = read_kinodb_sequential(kdb_path)
        seq_times["kinodb"].append(d)

    results["seq_read"] = {k: np.mean(v) for k, v in seq_times.items()}

    # ── Random read (averaged) ───────────────────────────────
    print("Random read...")
    rand_times = {"HDF5": [], "Parquet": [], "kinodb": []}
    for _ in range(args.runs):
        d, _ = read_hdf5_random(hdf5_path, random_indices)
        rand_times["HDF5"].append(d)
        d, _ = read_parquet_random(parquet_path, random_indices)
        rand_times["Parquet"].append(d)
        d, _ = read_kinodb_random(kdb_path, random_indices)
        rand_times["kinodb"].append(d)

    results["rand_read"] = {k: np.mean(v) for k, v in rand_times.items()}

    # ── Query (averaged) ─────────────────────────────────────
    print("Query (embodiment = 'franka')...")
    query_times = {"HDF5": [], "Parquet": [], "kinodb": []}
    for _ in range(args.runs):
        d, _, _ = query_hdf5(hdf5_path)
        query_times["HDF5"].append(d)
        d, _, _ = query_parquet(parquet_path)
        query_times["Parquet"].append(d)
        d, _, _ = query_kinodb(kdb_path)
        query_times["kinodb"].append(d)

    results["query"] = {k: np.mean(v) for k, v in query_times.items()}

    # ── Print results ────────────────────────────────────────
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print()

    def row(label, hdf5_val, pq_val, kdb_val, fmt_fn, lower_is_better=True):
        vals = [hdf5_val, pq_val, kdb_val]
        best = min(vals) if lower_is_better else max(vals)
        markers = ["*" if v == best else " " for v in vals]
        print(f"  {label:<24} {markers[0]}{fmt_fn(hdf5_val):<16} {markers[1]}{fmt_fn(pq_val):<16} {markers[2]}{fmt_fn(kdb_val):<16}")

    print(f"  {'Metric':<24} {'HDF5':<17} {'Parquet':<17} {'kinodb':<17}")
    print(f"  {'-'*24} {'-'*16} {'-'*16} {'-'*16}")

    row("File size", hdf5_size, parquet_size, kdb_size, format_bytes)
    row("Write time", results["write_time"]["HDF5"], results["write_time"]["Parquet"], results["write_time"]["kinodb"], format_time)
    row("Seq. read (all)", results["seq_read"]["HDF5"], results["seq_read"]["Parquet"], results["seq_read"]["kinodb"], format_time)
    row(f"Random read ({n_random})", results["rand_read"]["HDF5"], results["rand_read"]["Parquet"], results["rand_read"]["kinodb"], format_time)
    row("Query (filter)", results["query"]["HDF5"], results["query"]["Parquet"], results["query"]["kinodb"], format_time)

    print()
    print(f"  * = best in category")
    print()

    # ── Speedup summary ──────────────────────────────────────
    print("Speedup (kinodb vs others):")
    for metric in ["seq_read", "rand_read", "query"]:
        h = results[metric]["HDF5"] / results[metric]["kinodb"]
        p = results[metric]["Parquet"] / results[metric]["kinodb"]
        label = {"seq_read": "Sequential read", "rand_read": "Random read", "query": "Filtered query"}[metric]
        print(f"  {label:<24} {h:.1f}x vs HDF5, {p:.1f}x vs Parquet")

    size_h = hdf5_size / kdb_size
    size_p = parquet_size / kdb_size
    print(f"  {'File size':<24} {size_h:.1f}x vs HDF5, {size_p:.1f}x vs Parquet")

    # ── Save raw results ─────────────────────────────────────
    raw_path = os.path.join(tmpdir, "benchmark_results.json")
    with open(raw_path, "w") as f:
        json.dump({
            "config": {"episodes": n_ep, "frames": n_fr, "images": args.images, "runs": args.runs},
            "file_sizes": {k: v for k, v in results["file_size"].items()},
            "write_time_s": results["write_time"],
            "seq_read_s": results["seq_read"],
            "rand_read_s": results["rand_read"],
            "query_s": results["query"],
        }, f, indent=2)
    print()
    print(f"Raw results saved to: {raw_path}")
    print(f"Temp directory: {tmpdir}")
    print("  (delete with: rm -rf", tmpdir, ")")


if __name__ == "__main__":
    main()
