#!/usr/bin/env python3
"""
kinodb Real Dataset Benchmarks
==============================

Downloads actual robotics datasets and benchmarks kinodb against
native format reads (h5py for HDF5, pyarrow for LeRobot Parquet).

Datasets:
  1. LIBERO-Spatial (HDF5) — robomimic-style, from the LIBERO benchmark
  2. LeRobot pusht (Parquet) — from HuggingFace lerobot/pusht

Requirements:
  pip install h5py pyarrow numpy huggingface_hub
  cd crates/kinodb-py && maturin develop --release
  cargo build --release
  export PATH="$PWD/target/release:$PATH"

Usage:
  python benchmark_real.py
  python benchmark_real.py --runs 5
  python benchmark_real.py --skip-download   # if already downloaded
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

# ── Helpers ──────────────────────────────────────────────────

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

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARN: command failed: {cmd}")
        print(f"  stderr: {result.stderr[:500]}")
    return result.returncode == 0

def file_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total
    return 0

# ── Download datasets ────────────────────────────────────────

def download_libero(data_dir):
    """Download a publicly accessible robotics HDF5 dataset."""
    libero_dir = os.path.join(data_dir, "libero")
    if os.path.exists(libero_dir) and any(f.endswith('.hdf5') for f in os.listdir(libero_dir)):
        print("  LIBERO already downloaded, skipping.")
        return libero_dir

    os.makedirs(libero_dir, exist_ok=True)
    print("  Downloading LIBERO demo data...")

    # Try multiple sources
    sources = [
        # Public LIBERO spatial task
        ("yifengzhu-hf/LIBERO-datasets", "libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"),
        # Fallback: another public robomimic dataset
        ("jxu124/robomimic-image", "can/ph/image.hdf5"),
    ]

    try:
        from huggingface_hub import hf_hub_download
        for repo_id, filename in sources:
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=libero_dir,
                )
                print(f"  Downloaded: {path}")
                return libero_dir
            except Exception as e:
                print(f"  WARN: {repo_id} failed: {e}")
                continue
    except ImportError:
        print("  WARN: huggingface_hub not installed")

    print("  Could not download HDF5 data. Skipping.")
    return libero_dir


def download_lerobot(data_dir):
    """Download a LeRobot dataset from HuggingFace."""
    lerobot_dir = os.path.join(data_dir, "lerobot_pusht")
    if os.path.exists(lerobot_dir) and os.path.exists(os.path.join(lerobot_dir, "data")):
        print("  LeRobot pusht already downloaded, skipping.")
        return lerobot_dir

    os.makedirs(lerobot_dir, exist_ok=True)
    print("  Downloading LeRobot pusht from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "lerobot/pusht",
            repo_type="dataset",
            local_dir=lerobot_dir,
            allow_patterns=["data/**", "meta/**"],
        )
        print(f"  Downloaded to: {lerobot_dir}")
    except Exception as e:
        print(f"  WARN: LeRobot download failed: {e}")
    return lerobot_dir


# ── HDF5 native benchmarks ──────────────────────────────────

def bench_hdf5_native(hdf5_path, n_runs=3):
    """Benchmark native h5py reads on a LIBERO HDF5 file."""
    import h5py

    results = {}

    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys())
        n_episodes = len(demo_keys)
        total_frames = sum(data[k]["actions"].shape[0] for k in demo_keys)

    results["num_episodes"] = n_episodes
    results["total_frames"] = total_frames

    # Sequential read
    seq_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            for key in sorted(data.keys()):
                demo = data[key]
                actions = demo["actions"][:]
                obs = demo["obs"]
                # Read all state keys
                for obs_key in obs.keys():
                    ds = obs[obs_key]
                    if len(ds.shape) == 2:
                        _ = ds[:]
        seq_times.append(time.perf_counter() - t0)
    results["seq_read"] = np.mean(seq_times)

    # Random read (min(100, n_episodes) random episodes)
    n_random = min(100, n_episodes)
    indices = np.random.choice(n_episodes, n_random, replace=False).tolist()
    rand_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            keys = sorted(data.keys())
            for idx in indices:
                demo = data[keys[idx]]
                actions = demo["actions"][:]
                obs = demo["obs"]
                for obs_key in obs.keys():
                    ds = obs[obs_key]
                    if len(ds.shape) == 2:
                        _ = ds[:]
        rand_times.append(time.perf_counter() - t0)
    results["rand_read"] = np.mean(rand_times)
    results["n_random"] = n_random

    return results


# ── LeRobot Parquet native benchmarks ────────────────────────

def bench_parquet_native(parquet_dir, n_runs=3):
    """Benchmark native pyarrow reads on LeRobot Parquet files."""
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds

    # Find all parquet data files
    data_dir = os.path.join(parquet_dir, "data")
    pq_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".parquet"):
                pq_files.append(os.path.join(root, f))
    pq_files.sort()

    if not pq_files:
        return None

    results = {}

    # Read all parquet files into one table
    def read_all_parquets():
        if len(pq_files) == 1:
            return pq.read_table(pq_files[0]).to_pandas()
        else:
            tables = [pq.read_table(f) for f in pq_files]
            import pyarrow as pa
            combined = pa.concat_tables(tables)
            return combined.to_pandas()

    df = read_all_parquets()

    n_episodes = df["episode_index"].nunique()
    total_frames = len(df)
    results["num_episodes"] = n_episodes
    results["total_frames"] = total_frames

    # Find action and state column names
    action_cols = [c for c in df.columns if c == "action" or c.startswith("action.") or c.startswith("action_")]
    state_cols = [c for c in df.columns if "state" in c.lower() and "observation" in c.lower()]

    def extract_array(ep_df, cols):
        """Extract float array from columns, handling both scalar and list-type columns."""
        if not cols:
            return None
        col = cols[0]
        sample = ep_df[col].iloc[0] if len(ep_df) > 0 else None
        if sample is not None and hasattr(sample, '__len__') and not isinstance(sample, str):
            # List-type column: stack the lists into a 2D array
            return np.stack(ep_df[col].values)
        else:
            # Scalar columns: read as normal
            try:
                return ep_df[cols].values.astype(np.float32)
            except (ValueError, TypeError):
                return None

    # Sequential read (all episodes)
    seq_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        df = read_all_parquets()
        for ep_idx in range(n_episodes):
            ep_df = df[df["episode_index"] == ep_idx]
            _ = extract_array(ep_df, action_cols)
            _ = extract_array(ep_df, state_cols)
        seq_times.append(time.perf_counter() - t0)
    results["seq_read"] = np.mean(seq_times)

    # Random read
    n_random = min(100, n_episodes)
    indices = np.random.choice(n_episodes, n_random, replace=False).tolist()
    rand_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        df = read_all_parquets()
        for idx in indices:
            ep_df = df[df["episode_index"] == idx]
            _ = extract_array(ep_df, action_cols)
        rand_times.append(time.perf_counter() - t0)
    results["rand_read"] = np.mean(rand_times)
    results["n_random"] = n_random

    return results


# ── kinodb benchmarks ────────────────────────────────────────

def bench_kinodb(kdb_path, n_runs=3):
    """Benchmark kinodb reads via Python bindings."""
    import kinodb

    db = kinodb.open(kdb_path)
    n_episodes = db.num_episodes()

    results = {}
    results["num_episodes"] = n_episodes

    # Count total frames
    total_frames = 0
    for i in range(n_episodes):
        meta = db.read_meta(i)
        total_frames += meta["num_frames"]
    results["total_frames"] = total_frames

    # Sequential read
    seq_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        db2 = kinodb.open(kdb_path)
        for i in range(db2.num_episodes()):
            ep = db2.read_episode(i)
            _ = np.array(ep["actions"], dtype=np.float32)
            _ = np.array(ep["states"], dtype=np.float32)
        seq_times.append(time.perf_counter() - t0)
    results["seq_read"] = np.mean(seq_times)

    # Random read
    n_random = min(100, n_episodes)
    indices = np.random.choice(n_episodes, n_random, replace=False).tolist()
    rand_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        db2 = kinodb.open(kdb_path)
        for idx in indices:
            ep = db2.read_episode(idx)
            _ = np.array(ep["actions"], dtype=np.float32)
        rand_times.append(time.perf_counter() - t0)
    results["rand_read"] = np.mean(rand_times)
    results["n_random"] = n_random

    # KQL query
    query_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        db2 = kinodb.open(kdb_path)
        hits = db2.query("success = true")
        query_times.append(time.perf_counter() - t0)
    results["query"] = np.mean(query_times)
    results["query_hits"] = len(hits)

    # Metadata-only scan
    meta_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        db2 = kinodb.open(kdb_path)
        for i in range(db2.num_episodes()):
            _ = db2.read_meta(i)
        meta_times.append(time.perf_counter() - t0)
    results["meta_scan"] = np.mean(meta_times)

    return results


# ── Print results ────────────────────────────────────────────

def print_comparison(title, native_name, native_results, kdb_results, native_size, kdb_size):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print()
    print(f"  Episodes:     {kdb_results['num_episodes']}")
    print(f"  Total frames: {kdb_results['total_frames']}")
    print()

    print(f"  {'Metric':<28} {native_name:<20} {'kinodb':<20} {'Speedup':<15}")
    print(f"  {'-'*28} {'-'*19} {'-'*19} {'-'*14}")

    # File size
    print(f"  {'File size':<28} {format_bytes(native_size):<20} {format_bytes(kdb_size):<20} {native_size/max(kdb_size,1):.1f}x smaller")

    # Sequential read
    ns = native_results["seq_read"]
    ks = kdb_results["seq_read"]
    sp = ns / max(ks, 1e-9)
    print(f"  {'Sequential read (all)':<28} {format_time(ns):<20} {format_time(ks):<20} {sp:.1f}x faster")

    # Random read
    nr = native_results["rand_read"]
    kr = kdb_results["rand_read"]
    n_rand = kdb_results.get("n_random", 100)
    sp = nr / max(kr, 1e-9)
    print(f"  {'Random read (' + str(n_rand) + ' ep)':<28} {format_time(nr):<20} {format_time(kr):<20} {sp:.1f}x faster")

    # KQL query (kinodb only)
    if "query" in kdb_results:
        print(f"  {'KQL query':<28} {'N/A':<20} {format_time(kdb_results['query']):<20} (kinodb-only)")

    # Meta scan (kinodb only)
    if "meta_scan" in kdb_results:
        print(f"  {'Metadata scan':<28} {'N/A':<20} {format_time(kdb_results['meta_scan']):<20} (kinodb-only)")

    print()


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="kinodb real dataset benchmarks")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset downloads")
    parser.add_argument("--data-dir", default="./benchmark_data", help="Where to store downloaded data")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 72)
    print("  kinodb Real Dataset Benchmarks")
    print("=" * 72)
    print(f"  Runs: {args.runs}")
    print(f"  Data dir: {data_dir}")
    print()

    all_results = {}

    # ── Benchmark 1: LIBERO (HDF5) ──────────────────────────
    print("--- LIBERO-Spatial (HDF5) ---")

    if not args.skip_download:
        libero_dir = download_libero(data_dir)
    else:
        libero_dir = os.path.join(data_dir, "libero")

    # Find the HDF5 file
    hdf5_files = []
    for root, _, files in os.walk(libero_dir):
        for f in files:
            if f.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, f))

    if hdf5_files:
        hdf5_path = hdf5_files[0]
        kdb_path = os.path.join(data_dir, "libero.kdb")

        print(f"  HDF5 file: {hdf5_path}")
        hdf5_size = file_size(hdf5_path)
        print(f"  Size: {format_bytes(hdf5_size)}")

        # Ingest into kinodb
        print("  Ingesting into kinodb...")
        run_cmd(f"kino ingest \"{hdf5_path}\" --output \"{kdb_path}\" --embodiment franka --task libero_spatial --fps 20")
        kdb_size = file_size(kdb_path)

        # Benchmark native HDF5
        print(f"  Benchmarking h5py ({args.runs} runs)...")
        hdf5_results = bench_hdf5_native(hdf5_path, args.runs)

        # Benchmark kinodb
        print(f"  Benchmarking kinodb ({args.runs} runs)...")
        kdb_results = bench_kinodb(kdb_path, args.runs)

        print_comparison(
            "LIBERO-Spatial: HDF5 vs kinodb",
            "HDF5 (h5py)",
            hdf5_results,
            kdb_results,
            hdf5_size,
            kdb_size,
        )
        all_results["libero"] = {
            "hdf5": hdf5_results,
            "kinodb": kdb_results,
            "hdf5_size": hdf5_size,
            "kdb_size": kdb_size,
        }
    else:
        print("  WARN: No LIBERO HDF5 files found. Skipping.")

    # ── Benchmark 2: LeRobot pusht (Parquet) ─────────────────
    print()
    print("--- LeRobot pusht (Parquet) ---")

    if not args.skip_download:
        lerobot_dir = download_lerobot(data_dir)
    else:
        lerobot_dir = os.path.join(data_dir, "lerobot_pusht")

    data_subdir = os.path.join(lerobot_dir, "data")
    if os.path.exists(data_subdir):
        kdb_path_lr = os.path.join(data_dir, "lerobot_pusht.kdb")
        lr_size = file_size(lerobot_dir)
        print(f"  Dataset dir: {lerobot_dir}")
        print(f"  Size: {format_bytes(lr_size)}")

        # Ingest into kinodb
        print("  Ingesting into kinodb...")
        run_cmd(f"kino ingest \"{lerobot_dir}\" --format lerobot --output \"{kdb_path_lr}\"")
        kdb_lr_size = file_size(kdb_path_lr)

        # Benchmark native Parquet
        print(f"  Benchmarking pyarrow ({args.runs} runs)...")
        pq_results = bench_parquet_native(lerobot_dir, args.runs)

        if pq_results:
            # Benchmark kinodb
            print(f"  Benchmarking kinodb ({args.runs} runs)...")
            kdb_lr_results = bench_kinodb(kdb_path_lr, args.runs)

            print_comparison(
                "LeRobot pusht: Parquet vs kinodb",
                "Parquet (pyarrow)",
                pq_results,
                kdb_lr_results,
                lr_size,
                kdb_lr_size,
            )
            all_results["lerobot_pusht"] = {
                "parquet": pq_results,
                "kinodb": kdb_lr_results,
                "parquet_size": lr_size,
                "kdb_size": kdb_lr_size,
            }
        else:
            print("  WARN: Could not read Parquet files. Skipping.")
    else:
        print("  WARN: LeRobot data/ not found. Skipping.")

    # ── Save results ─────────────────────────────────────────
    results_path = os.path.join(data_dir, "real_benchmark_results.json")

    # Convert numpy types to Python native for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print()
    print(f"Results saved to: {results_path}")
    print(f"Data directory: {data_dir}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()