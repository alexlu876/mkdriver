"""Sequence-returning BC dataset + demo-aware batch sampler.

Per MKW_RL_SPEC.md §1.6 / §2.3:

* The dataset returns fixed-length sequences (``seq_len`` timesteps) for
  TBPTT. Each timestep carries a frame stack of ``stack_size`` past
  frames (with frameskip), producing shape ``(T, stack, H, W)``.
* Demo boundaries are never crossed: a sequence is always contiguous
  within a single demo.
* The DemoAwareBatchSampler distributes demos across batch positions
  such that within a batch position, the sampler yields windows of one
  demo in timestep order, then moves on to the next demo assigned to
  that position. This lets the TBPTT training loop carry LSTM hidden
  state across batches within a demo, and reset it at demo boundaries.

Frame stacking:
  For timestep ``t`` the observation is
      [frames[t - 3*frameskip], frames[t - 2*frameskip],
       frames[t - frameskip],   frames[t]]
  Pad with copies of frame 0 of the same demo when the stack extends
  before the demo's start. Never pad across a demo boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .action_encoding import encode_steering
from .frames import load_frame
from .pairing import PairedSample

if TYPE_CHECKING:
    pass  # no conditional imports currently needed

log = logging.getLogger(__name__)

DEFAULT_FRAME_SIZE: tuple[int, int] = (140, 75)  # (width, height) — matches VIPTankz's MKWii BTR impl


# ---------------------------------------------------------------------------
# Per-demo data container.
# ---------------------------------------------------------------------------


@dataclass
class DemoData:
    """All paired samples for one demo plus its chunk index.

    Attributes:
        demo_id: Identifier (usually the demo's source filename stem).
        samples: Paired samples in emission order.
        chunk_starts: List of starting timestep indices for each
            non-overlapping chunk of length ``seq_len``. Excludes any
            trailing partial chunk.
    """

    demo_id: str
    samples: list[PairedSample]
    chunk_starts: list[int] = field(default_factory=list)

    def n_chunks(self) -> int:
        return len(self.chunk_starts)


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------


@dataclass
class _ChunkAddress:
    demo_id: str
    seq_start: int


class MkwBCDataset(Dataset):
    """Flat index over all chunks across all demos.

    ``__getitem__(idx)`` returns a dict:
        {
            "frames":   torch.Tensor (T, stack_size, H, W), float32 in [0,1]
            "actions": {
                "steering_bin": torch.Tensor (T,) long
                "accelerate":   torch.Tensor (T,) float
                "brake":        torch.Tensor (T,) float
                "drift":        torch.Tensor (T,) float
                "item":         torch.Tensor (T,) float
            },
            "meta": {
                "demo_id":     str
                "seq_start":   int   # input_frame_idx of the chunk's first timestep, within the demo
            }
        }

    ``is_continuation`` is deliberately NOT in the dataset output — it
    depends on what the sampler did for this batch position last batch.
    The training loop computes it by comparing ``meta["demo_id"]`` and
    ``meta["seq_start"]`` across successive batches.
    """

    def __init__(
        self,
        samples_by_demo: dict[str, list[PairedSample]],
        stack_size: int = 4,
        frame_skip: int = 4,
        seq_len: int = 32,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
    ) -> None:
        if stack_size < 1:
            raise ValueError("stack_size must be >= 1")
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if not samples_by_demo:
            raise ValueError("samples_by_demo is empty")

        self.stack_size = stack_size
        self.frame_skip = frame_skip
        self.seq_len = seq_len
        self.frame_size = frame_size

        self._demos: dict[str, DemoData] = {}
        self._chunk_addresses: list[_ChunkAddress] = []

        for demo_id, samples in samples_by_demo.items():
            if len(samples) < seq_len:
                log.warning(
                    "demo %s has %d samples (< seq_len=%d); skipping",
                    demo_id,
                    len(samples),
                    seq_len,
                )
                continue
            n_chunks = len(samples) // seq_len
            if n_chunks == 0:
                continue
            starts = [i * seq_len for i in range(n_chunks)]
            demo = DemoData(demo_id=demo_id, samples=samples, chunk_starts=starts)
            self._demos[demo_id] = demo
            for s in starts:
                self._chunk_addresses.append(_ChunkAddress(demo_id=demo_id, seq_start=s))

        if not self._chunk_addresses:
            raise ValueError("no demos with enough samples for even one chunk of seq_len")

    def __len__(self) -> int:
        return len(self._chunk_addresses)

    @property
    def demo_ids(self) -> list[str]:
        return list(self._demos.keys())

    def demo(self, demo_id: str) -> DemoData:
        return self._demos[demo_id]

    def chunks_for_demo(self, demo_id: str) -> list[int]:
        """Flat chunk indices (into the dataset) that belong to this demo, in order."""
        return [i for i, c in enumerate(self._chunk_addresses) if c.demo_id == demo_id]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        addr = self._chunk_addresses[idx]
        demo = self._demos[addr.demo_id]
        start = addr.seq_start
        end = start + self.seq_len

        # ---- Frame stacks per timestep ----
        # For each t in [start, end), build a stack of self.stack_size past
        # frames at self.frame_skip intervals. Pad with the demo's first
        # frame when the stack extends before the demo.
        H = self.frame_size[1]
        W = self.frame_size[0]
        frames = np.empty((self.seq_len, self.stack_size, H, W), dtype=np.float32)
        for t_rel, t in enumerate(range(start, end)):
            for s in range(self.stack_size):
                offset = (self.stack_size - 1 - s) * self.frame_skip
                src_t = max(0, t - offset)
                arr = load_frame(
                    demo.samples[src_t].frame_path,
                    size=self.frame_size,
                    grayscale=True,
                )
                frames[t_rel, s] = arr.astype(np.float32) / 255.0

        frames_t = torch.from_numpy(frames)

        # ---- Per-timestep actions ----
        T = self.seq_len
        steering = np.empty(T, dtype=np.int64)
        accelerate = np.empty(T, dtype=np.float32)
        brake = np.empty(T, dtype=np.float32)
        drift = np.empty(T, dtype=np.float32)
        item = np.empty(T, dtype=np.float32)
        for t_rel, t in enumerate(range(start, end)):
            c = demo.samples[t].controller
            steering[t_rel] = encode_steering(c.steering)
            accelerate[t_rel] = 1.0 if c.accelerate else 0.0
            brake[t_rel] = 1.0 if c.brake else 0.0
            drift[t_rel] = 1.0 if c.drift else 0.0
            item[t_rel] = 1.0 if c.item else 0.0

        return {
            "frames": frames_t,
            "actions": {
                "steering_bin": torch.from_numpy(steering),
                "accelerate": torch.from_numpy(accelerate),
                "brake": torch.from_numpy(brake),
                "drift": torch.from_numpy(drift),
                "item": torch.from_numpy(item),
            },
            "meta": {
                "demo_id": addr.demo_id,
                "seq_start": start,
            },
        }


# ---------------------------------------------------------------------------
# Demo-aware batch sampler.
# ---------------------------------------------------------------------------


class DemoAwareBatchSampler(Sampler[list[int]]):
    """Distribute demos across ``batch_size`` streams and emit lockstep batches.

    For batch position ``b`` the sampler yields chunks from the demos in
    ``streams[b]`` in order: demo_0 chunk_0, demo_0 chunk_1, ..., demo_0
    chunk_N, demo_1 chunk_0, demo_1 chunk_1, ...

    Once any stream is exhausted the epoch ends (so batches stay
    full-width). This does trim some chunks from longer streams; use
    ``drop_last=True`` downstream.

    Demo-to-stream assignment is round-robin over demos sorted by
    length (descending), giving ~balanced streams. Between epochs,
    the per-stream order of demos is shuffled (but the within-demo
    chunk order is preserved).
    """

    def __init__(
        self,
        dataset: MkwBCDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._streams: list[list[int]] = self._build_streams()

    def _build_streams(self) -> list[list[int]]:
        demo_ids = self.dataset.demo_ids
        # Sort demos by number of chunks (descending) for better balance.
        demo_ids_sorted = sorted(
            demo_ids,
            key=lambda d: self.dataset.demo(d).n_chunks(),
            reverse=True,
        )
        streams: list[list[str]] = [[] for _ in range(self.batch_size)]
        for i, did in enumerate(demo_ids_sorted):
            streams[i % self.batch_size].append(did)

        if self.shuffle:
            for s in streams:
                # Shuffle within-stream demo order (but not chunk order within demo).
                self._rng.shuffle(s)

        # Flatten each stream to a list of chunk indices.
        stream_chunk_indices: list[list[int]] = []
        for s in streams:
            flat: list[int] = []
            for did in s:
                flat.extend(self.dataset.chunks_for_demo(did))
            stream_chunk_indices.append(flat)

        return stream_chunk_indices

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            self._streams = self._build_streams()

        n_batches = min((len(s) for s in self._streams), default=0)
        for b_idx in range(n_batches):
            yield [self._streams[p][b_idx] for p in range(self.batch_size)]

    def __len__(self) -> int:
        return min((len(s) for s in self._streams), default=0)


# ---------------------------------------------------------------------------
# Utilities for the training loop.
# ---------------------------------------------------------------------------


def compute_is_continuation(
    prev_meta: dict[str, Any] | None,
    curr_meta: dict[str, Any],
    seq_len: int,
) -> bool:
    """Was ``curr_meta`` the next chunk after ``prev_meta`` in the same demo?

    Used by the training loop to decide whether to carry LSTM hidden
    state or reset it for a given batch position. Returns False if
    prev is None (first batch of epoch).
    """
    if prev_meta is None:
        return False
    if prev_meta["demo_id"] != curr_meta["demo_id"]:
        return False
    expected_next = prev_meta["seq_start"] + seq_len
    return curr_meta["seq_start"] == expected_next


# ---------------------------------------------------------------------------
# Collate: stack batch-position items into a single nested dict of tensors.
# ---------------------------------------------------------------------------


def bc_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate list-of-dicts into a dict of stacked tensors.

    Shapes after collate:
        frames:               (B, T, stack, H, W)
        actions[steering_bin]:(B, T)
        actions[accelerate]:  (B, T)  [etc.]
        meta[demo_id]:        list of B strings
        meta[seq_start]:      list of B ints
    """
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    actions: dict[str, torch.Tensor] = {}
    for key in batch[0]["actions"]:
        actions[key] = torch.stack([b["actions"][key] for b in batch], dim=0)
    meta = {
        "demo_id": [b["meta"]["demo_id"] for b in batch],
        "seq_start": [b["meta"]["seq_start"] for b in batch],
    }
    return {"frames": frames, "actions": actions, "meta": meta}


# ---------------------------------------------------------------------------
# Helpers to construct a dataset from scripts.
# ---------------------------------------------------------------------------


def demo_id_from_path(p: Path | str) -> str:
    """Derive a stable demo_id from a .dtm filename (stem)."""
    return Path(p).stem
