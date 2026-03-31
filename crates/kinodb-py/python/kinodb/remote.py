"""Python client for kinodb gRPC server.

Usage:
    from kinodb.remote import KinoClient, KinoRemoteDataset

    # Direct client
    client = KinoClient("localhost:50051")
    info = client.server_info()
    ep = client.get_episode(0)
    batch = client.get_batch(32, mode="random")
    hits = client.query("embodiment = 'franka' AND success = true")

    # PyTorch DataLoader
    dataset = KinoRemoteDataset("localhost:50051", batch_size=32)
    for batch in dataset:
        actions = batch["action"]  # (B, T, action_dim)
"""

from __future__ import annotations

import numpy as np

try:
    import grpc
    from google.protobuf import descriptor_pool
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class KinoClient:
    """Client for the kinodb gRPC server.

    Args:
        address: Server address, e.g. "localhost:50051"
        timeout: RPC timeout in seconds.
    """

    def __init__(self, address: str = "localhost:50051", timeout: float = 30.0):
        if not HAS_GRPC:
            raise ImportError(
                "grpcio and protobuf are required. Install with: "
                "pip install grpcio grpcio-tools protobuf"
            )

        self.address = address
        self.timeout = timeout
        self.channel = grpc.insecure_channel(address)

        # Import generated stubs (user must generate these from kinodb.proto)
        # For now, use a raw channel approach
        self._connected = False

    def _ensure_stubs(self):
        """Lazy-load gRPC stubs."""
        if not self._connected:
            try:
                from kinodb import kinodb_pb2, kinodb_pb2_grpc
                self._stub = kinodb_pb2_grpc.KinoServiceStub(self.channel)
                self._pb2 = kinodb_pb2
                self._connected = True
            except ImportError:
                raise ImportError(
                    "gRPC stubs not generated. Run:\n"
                    "  python -m grpc_tools.protoc -I proto "
                    "--python_out=python/kinodb --grpc_python_out=python/kinodb "
                    "proto/kinodb.proto"
                )

    def server_info(self) -> dict:
        """Get server info (episodes, frames, sources)."""
        self._ensure_stubs()
        resp = self._stub.ServerInfo(
            self._pb2.InfoRequest(),
            timeout=self.timeout,
        )
        return {
            "num_episodes": resp.num_episodes,
            "num_frames": resp.num_frames,
            "num_sources": resp.num_sources,
            "version": resp.version,
            "sources": list(resp.sources),
        }

    def get_episode(self, position: int) -> dict:
        """Get a full episode by position."""
        self._ensure_stubs()
        resp = self._stub.GetEpisode(
            self._pb2.EpisodeRequest(position=position),
            timeout=self.timeout,
        )
        return _episode_to_dict(resp.episode)

    def get_meta(self, position: int) -> dict:
        """Get episode metadata (cheap, no frame data)."""
        self._ensure_stubs()
        resp = self._stub.GetMeta(
            self._pb2.MetaRequest(position=position),
            timeout=self.timeout,
        )
        return _meta_to_dict(resp.meta)

    def get_batch(
        self,
        batch_size: int = 32,
        mode: str = "random",
        offset: int = 0,
        include_images: bool = False,
    ) -> list[dict]:
        """Get a batch of episodes."""
        self._ensure_stubs()
        resp = self._stub.GetBatch(
            self._pb2.BatchRequest(
                batch_size=batch_size,
                mode=mode,
                offset=offset,
                include_images=include_images,
            ),
            timeout=self.timeout,
        )
        return [_episode_to_dict(ep) for ep in resp.episodes]

    def query(self, kql: str, limit: int = 0) -> list[dict]:
        """Filter episodes with KQL. Returns list of {position, meta}."""
        self._ensure_stubs()
        resp = self._stub.Query(
            self._pb2.QueryRequest(kql=kql, limit=limit),
            timeout=self.timeout,
        )
        results = []
        for pos, meta in zip(resp.positions, resp.metas):
            d = _meta_to_dict(meta)
            d["position"] = pos
            results.append(d)
        return results

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _meta_to_dict(meta) -> dict:
    d = {
        "episode_id": meta.episode_id,
        "embodiment": meta.embodiment,
        "task": meta.task,
        "num_frames": meta.num_frames,
        "fps": meta.fps,
        "action_dim": meta.action_dim,
    }
    if meta.has_success:
        d["success"] = meta.success
    if meta.has_reward:
        d["total_reward"] = meta.total_reward
    return d


def _episode_to_dict(ep) -> dict:
    meta = _meta_to_dict(ep.meta)
    actions = np.array([f.action for f in ep.frames], dtype=np.float32)
    states = np.array([f.state for f in ep.frames], dtype=np.float32)
    rewards = np.array([f.reward for f in ep.frames], dtype=np.float32)
    terminals = [f.is_terminal for f in ep.frames]

    result = {
        "meta": meta,
        "action": actions,
        "state": states,
        "reward": rewards,
        "is_terminal": terminals,
        "task": meta["task"],
        "embodiment": meta["embodiment"],
    }

    # Images (if present)
    if ep.frames and ep.frames[0].images:
        cam_name = ep.frames[0].images[0].camera
        frames = []
        for f in ep.frames:
            for img in f.images:
                if img.camera == cam_name:
                    h, w, c = img.height, img.width, img.channels
                    pixels = np.frombuffer(img.data, dtype=np.uint8).reshape(h, w, c)
                    frames.append(pixels)
                    break
        if frames:
            images = np.stack(frames, axis=0).astype(np.float32) / 255.0
            images = np.transpose(images, (0, 3, 1, 2))  # (T, C, H, W)
            result["image"] = images
            result["camera"] = cam_name

    return result


class KinoRemoteDataset:
    """PyTorch IterableDataset that streams batches from kino-serve.

    Args:
        address: Server address, e.g. "localhost:50051"
        batch_size: Episodes per batch.
        include_images: Whether to transfer image data.
        to_tensor: Convert numpy arrays to torch tensors.
    """

    def __init__(
        self,
        address: str = "localhost:50051",
        batch_size: int = 32,
        include_images: bool = False,
        to_tensor: bool = True,
    ):
        self.client = KinoClient(address)
        self.batch_size = batch_size
        self.include_images = include_images
        self.to_tensor = to_tensor and HAS_TORCH

        info = self.client.server_info()
        self._num_episodes = info["num_episodes"]

    def __len__(self):
        return self._num_episodes

    def __iter__(self):
        """Yield individual episodes from random batches."""
        n_batches = (self._num_episodes + self.batch_size - 1) // self.batch_size
        for _ in range(n_batches):
            episodes = self.client.get_batch(
                self.batch_size,
                mode="random",
                include_images=self.include_images,
            )
            for ep in episodes:
                if self.to_tensor:
                    ep = _to_tensors(ep)
                yield ep


def _to_tensors(sample: dict) -> dict:
    if not HAS_TORCH:
        return sample
    out = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        else:
            out[k] = v
    return out
