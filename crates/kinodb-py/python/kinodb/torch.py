"""PyTorch integration for kinodb.

Usage:
    from kinodb.torch import KinoDataset

    dataset = KinoDataset(
        "data.kdb",
        image_size=(224, 224),  # optional resize
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    for batch in loader:
        images = batch["image"]          # (B, C, H, W) float32
        actions = batch["action"]        # (B, action_dim) float32
        states = batch["state"]          # (B, state_dim) float32
        tasks = batch["task"]            # list of strings

Works with single .kdb files or with weighted mixtures:

    dataset = KinoDataset.from_mixture(
        sources={"bridge.kdb": 0.4, "aloha.kdb": 0.6},
        seed=42,
    )
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import kinodb as _kino


class KinoMapDataset:
    """A map-style dataset that loads episodes by index.

    Use this when you want random access (e.g. with a sampler)
    or when your dataset fits in memory.

    Args:
        db_path: Path to a .kdb file.
        kql_filter: Optional KQL query to filter episodes.
        image_key: Which camera to use (first camera if None).
        image_size: Optional (H, W) to resize images.
        to_tensor: If True, convert numpy arrays to torch tensors.
    """

    def __init__(
        self,
        db_path: str,
        kql_filter: Optional[str] = None,
        image_key: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        to_tensor: bool = True,
    ):
        self.db = _kino.open(db_path)
        self.image_key = image_key
        self.image_size = image_size
        self.to_tensor = to_tensor and HAS_TORCH

        # Apply filter if given
        if kql_filter:
            self.indices = self.db.query(kql_filter)
        else:
            self.indices = list(range(self.db.num_episodes()))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ep_pos = self.indices[idx]
        ep = self.db.read_episode(ep_pos)
        return self._episode_to_sample(ep)

    def _episode_to_sample(self, ep: dict) -> Dict[str, object]:
        """Convert a raw kinodb episode dict into a training-ready sample."""
        meta = ep["meta"]

        # Actions: (num_frames, action_dim) float32
        actions = np.array(ep["actions"], dtype=np.float32)

        # States: (num_frames, state_dim) float32
        states = np.array(ep["states"], dtype=np.float32)

        # Rewards: (num_frames,) float32
        rewards = np.array(ep["rewards"], dtype=np.float32)

        result = {
            "action": actions,
            "state": states,
            "reward": rewards,
            "task": meta["task"],
            "embodiment": meta["embodiment"],
            "success": meta.get("success"),
            "episode_id": meta["episode_id"],
        }

        # Images: pick one camera, stack across frames
        if ep["images"] and ep["images"][0]:
            cam_name = self.image_key
            if cam_name is None:
                # Use first camera
                cam_name = ep["images"][0][0]["camera"]

            frames = []
            for frame_imgs in ep["images"]:
                for img in frame_imgs:
                    if img["camera"] == cam_name:
                        h, w, c = img["height"], img["width"], img["channels"]
                        pixels = np.frombuffer(img["data"], dtype=np.uint8).reshape(h, w, c)
                        frames.append(pixels)
                        break

            if frames:
                # (num_frames, H, W, C)
                images = np.stack(frames, axis=0)

                # Optional resize
                if self.image_size is not None:
                    images = self._resize_batch(images, self.image_size)

                # Convert to (num_frames, C, H, W) float32 [0, 1]
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))

                result["image"] = images
                result["camera"] = cam_name

        # Convert to tensors if torch is available
        if self.to_tensor:
            result = self._to_tensors(result)

        return result

    @staticmethod
    def _resize_batch(images: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Simple nearest-neighbor resize. No PIL/cv2 dependency."""
        n, old_h, old_w, c = images.shape
        new_h, new_w = size
        row_idx = (np.arange(new_h) * old_h // new_h).astype(int)
        col_idx = (np.arange(new_w) * old_w // new_w).astype(int)
        return images[:, row_idx[:, None], col_idx[None, :], :]

    @staticmethod
    def _to_tensors(sample: dict) -> dict:
        """Convert numpy arrays to torch tensors, leave strings alone."""
        if not HAS_TORCH:
            return sample
        out = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)
            else:
                out[k] = v
        return out


class KinoIterDataset:
    """An iterable dataset that streams episodes, optionally with weighted mixing.

    Use this for large datasets that don't fit in memory, or for
    training with weighted mixtures of multiple .kdb files.

    Args:
        db_paths: Single path or dict of {path: weight}.
        kql_filter: Optional KQL query (only for single db).
        image_key: Which camera to use.
        image_size: Optional (H, W) resize.
        shuffle: Whether to shuffle episode order each epoch.
        seed: Random seed for reproducibility.
        to_tensor: Convert to torch tensors.
    """

    def __init__(
        self,
        db_paths,
        kql_filter: Optional[str] = None,
        image_key: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        shuffle: bool = True,
        seed: int = 42,
        to_tensor: bool = True,
    ):
        self.image_key = image_key
        self.image_size = image_size
        self.shuffle = shuffle
        self.seed = seed
        self.to_tensor = to_tensor and HAS_TORCH
        self._epoch = 0

        # Build source list: [(db, indices, weight)]
        self.sources: List[Tuple[object, List[int], float]] = []

        if isinstance(db_paths, str):
            db = _kino.open(db_paths)
            if kql_filter:
                indices = db.query(kql_filter)
            else:
                indices = list(range(db.num_episodes()))
            self.sources.append((db, indices, 1.0))
        elif isinstance(db_paths, dict):
            total_w = sum(db_paths.values())
            for path, weight in db_paths.items():
                db = _kino.open(path)
                indices = list(range(db.num_episodes()))
                self.sources.append((db, indices, weight / total_w))
        else:
            raise ValueError("db_paths must be a string or dict of {path: weight}")

        self._total = sum(len(idx) for _, idx, _ in self.sources)

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        """Yield episodes, respecting weights for multi-source mixtures."""
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        if len(self.sources) == 1:
            # Single source: simple shuffle
            db, indices, _ = self.sources[0]
            order = indices.copy()
            if self.shuffle:
                rng.shuffle(order)
            for pos in order:
                ep = db.read_episode(pos)
                yield self._episode_to_sample(ep)
        else:
            # Multi-source: interleave according to weights
            # Build a schedule: for each of _total samples, pick a source
            schedule = []
            for src_idx, (db, indices, weight) in enumerate(self.sources):
                n_from_src = max(1, round(weight * self._total))
                order = indices.copy()
                if self.shuffle:
                    rng.shuffle(order)
                # Cycle if n_from_src > len(indices)
                for i in range(n_from_src):
                    schedule.append((src_idx, order[i % len(order)]))

            if self.shuffle:
                rng.shuffle(schedule)

            for src_idx, ep_pos in schedule:
                db = self.sources[src_idx][0]
                ep = db.read_episode(ep_pos)
                yield self._episode_to_sample(ep)

    def _episode_to_sample(self, ep: dict) -> Dict[str, object]:
        """Same conversion as KinoMapDataset."""
        meta = ep["meta"]
        actions = np.array(ep["actions"], dtype=np.float32)
        states = np.array(ep["states"], dtype=np.float32)
        rewards = np.array(ep["rewards"], dtype=np.float32)

        result = {
            "action": actions,
            "state": states,
            "reward": rewards,
            "task": meta["task"],
            "embodiment": meta["embodiment"],
            "success": meta.get("success"),
            "episode_id": meta["episode_id"],
        }

        if ep["images"] and ep["images"][0]:
            cam_name = self.image_key or ep["images"][0][0]["camera"]
            frames = []
            for frame_imgs in ep["images"]:
                for img in frame_imgs:
                    if img["camera"] == cam_name:
                        h, w, c = img["height"], img["width"], img["channels"]
                        pixels = np.frombuffer(img["data"], dtype=np.uint8).reshape(h, w, c)
                        frames.append(pixels)
                        break
            if frames:
                images = np.stack(frames, axis=0)
                if self.image_size is not None:
                    images = KinoMapDataset._resize_batch(images, self.image_size)
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))
                result["image"] = images
                result["camera"] = cam_name

        if self.to_tensor:
            result = KinoMapDataset._to_tensors(result)

        return result


# ── Convenience aliases ──────────────────────────────────────

# Main public API
KinoDataset = KinoMapDataset


def from_mixture(
    sources: Dict[str, float],
    seed: int = 42,
    **kwargs,
) -> KinoIterDataset:
    """Create a weighted mixture dataset from multiple .kdb files.

    Args:
        sources: Dict of {path: weight}, e.g. {"bridge.kdb": 0.4, "aloha.kdb": 0.6}
        seed: Random seed.
        **kwargs: Passed to KinoIterDataset (image_key, image_size, etc.)

    Returns:
        KinoIterDataset that yields episodes weighted by the given ratios.
    """
    return KinoIterDataset(sources, seed=seed, **kwargs)
