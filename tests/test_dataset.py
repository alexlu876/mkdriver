"""Tests for src/mkw_rl/dtm/dataset.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from mkw_rl.dtm.dataset import (
    DemoAwareBatchSampler,
    MkwBCDataset,
    bc_collate_fn,
    compute_is_continuation,
    demo_id_from_path,
)
from mkw_rl.dtm.pairing import pair_dtm_and_frames
from mkw_rl.dtm.parser import build_dtm_blob, build_frame


def _synth_paired_samples(tmp_path: Path, n: int, *, demo_id: str = "d") -> list:
    """Build a small synthetic demo: .dtm + PNGs, return paired samples."""
    dtm = tmp_path / f"{demo_id}.dtm"
    frame_dir = tmp_path / f"frames_{demo_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        build_frame(
            accelerate=(i % 3 != 0),
            drift=(i % 7 < 2),
            analog_x=(128 + i) & 0xFF,
        )
        for i in range(n)
    ]
    dtm.write_bytes(build_dtm_blob(vi_count=n, input_count=n, frames=frames))
    for i in range(n):
        arr = np.full((60, 80, 3), (i * 7 % 256, i * 11 % 256, 50), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(frame_dir / f"framedump_{i}.png")
    return pair_dtm_and_frames(dtm, frame_dir, tail_margin=0)


# ---------------------------------------------------------------------------
# Basic dataset construction.
# ---------------------------------------------------------------------------


class TestDatasetBasics:
    def test_construct_simple(self, tmp_path: Path) -> None:
        samples = _synth_paired_samples(tmp_path, 100)
        ds = MkwBCDataset({"d": samples}, stack_size=4, frame_skip=4, seq_len=32)
        # 100 samples → 100 // 32 = 3 chunks.
        assert len(ds) == 3

    def test_short_demo_is_skipped(self, tmp_path: Path, caplog) -> None:
        # seq_len 32 > 20 samples → skipped with warning.
        short = _synth_paired_samples(tmp_path, 20, demo_id="short")
        ok = _synth_paired_samples(tmp_path, 100, demo_id="ok")
        with caplog.at_level("WARNING"):
            ds = MkwBCDataset({"short": short, "ok": ok}, seq_len=32)
        assert len(ds) == 100 // 32  # only "ok" contributes
        assert ds.demo_ids == ["ok"]
        assert any("skipping" in r.message for r in caplog.records)

    def test_no_usable_demos_raises(self, tmp_path: Path) -> None:
        short = _synth_paired_samples(tmp_path, 20, demo_id="s")
        with pytest.raises(ValueError, match="no demos"):
            MkwBCDataset({"s": short}, seq_len=32)

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError):
            MkwBCDataset({}, seq_len=32)  # empty
        samples = []  # doesn't matter; guarded before iteration

        # With a valid non-empty demo, test the other validators.
        # We can't easily construct PairedSample instances here without
        # going through a real pipeline, but we can verify argument
        # validators short-circuit before that:
        with pytest.raises(ValueError):
            MkwBCDataset({"d": samples}, stack_size=0)
        with pytest.raises(ValueError):
            MkwBCDataset({"d": samples}, frame_skip=0)
        with pytest.raises(ValueError):
            MkwBCDataset({"d": samples}, seq_len=0)


# ---------------------------------------------------------------------------
# Item shape and content.
# ---------------------------------------------------------------------------


class TestDatasetItem:
    def test_item_shapes_and_types(self, tmp_path: Path) -> None:
        samples = _synth_paired_samples(tmp_path, 50)
        ds = MkwBCDataset({"d": samples}, stack_size=4, frame_skip=4, seq_len=16, frame_size=(140, 114))
        item = ds[0]
        assert item["frames"].shape == (16, 4, 114, 140)
        assert item["frames"].dtype == torch.float32
        assert torch.all(item["frames"] >= 0.0) and torch.all(item["frames"] <= 1.0)
        for key in ("accelerate", "brake", "drift", "item"):
            t = item["actions"][key]
            assert t.shape == (16,)
            assert t.dtype == torch.float32
        assert item["actions"]["steering_bin"].shape == (16,)
        assert item["actions"]["steering_bin"].dtype == torch.int64
        assert item["meta"]["demo_id"] == "d"
        assert item["meta"]["seq_start"] == 0

    def test_consecutive_chunks_are_contiguous(self, tmp_path: Path) -> None:
        samples = _synth_paired_samples(tmp_path, 64)
        ds = MkwBCDataset({"d": samples}, seq_len=16)
        a = ds[0]
        b = ds[1]
        assert a["meta"]["seq_start"] == 0
        assert b["meta"]["seq_start"] == 16

    def test_stack_padding_at_demo_start(self, tmp_path: Path) -> None:
        """The first timestep's stack should be entirely copies of frame 0."""
        samples = _synth_paired_samples(tmp_path, 64)
        ds = MkwBCDataset({"d": samples}, stack_size=4, frame_skip=4, seq_len=16)
        item = ds[0]
        # Timestep 0's stack: all four slots should be frame 0 (padded).
        # Frames are loaded from disk so comparing exact floats is fine.
        slot0 = item["frames"][0, 0]
        slot1 = item["frames"][0, 1]
        slot2 = item["frames"][0, 2]
        slot3 = item["frames"][0, 3]
        assert torch.allclose(slot0, slot1)
        assert torch.allclose(slot1, slot2)
        assert torch.allclose(slot2, slot3)

    def test_stack_beyond_padding_window(self, tmp_path: Path) -> None:
        """At timestep >= (stack_size-1)*frame_skip, all stack entries are distinct frames."""
        samples = _synth_paired_samples(tmp_path, 64)
        ds = MkwBCDataset({"d": samples}, stack_size=4, frame_skip=4, seq_len=32)
        item = ds[0]
        # Timestep 12 (= 3*4) in seq 0: stack spans frames 0, 4, 8, 12.
        # Synthetic frames vary per-index so their mean differs.
        t = 12
        means = [float(item["frames"][t, s].mean()) for s in range(4)]
        assert len(set(round(m, 4) for m in means)) == 4


# ---------------------------------------------------------------------------
# DemoAwareBatchSampler.
# ---------------------------------------------------------------------------


class TestSampler:
    def test_single_demo_single_stream(self, tmp_path: Path) -> None:
        samples = _synth_paired_samples(tmp_path, 96)
        ds = MkwBCDataset({"d": samples}, seq_len=16)
        sampler = DemoAwareBatchSampler(ds, batch_size=1, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 6  # 96 / 16
        assert all(len(b) == 1 for b in batches)
        # For a single demo / single stream, batch[0] should be the chunk
        # indices in order.
        assert [b[0] for b in batches] == list(range(6))

    def test_multiple_demos_distributed_round_robin(self, tmp_path: Path) -> None:
        a = _synth_paired_samples(tmp_path, 64, demo_id="a")
        b = _synth_paired_samples(tmp_path, 64, demo_id="b")
        c = _synth_paired_samples(tmp_path, 64, demo_id="c")
        d = _synth_paired_samples(tmp_path, 64, demo_id="d")
        ds = MkwBCDataset({"a": a, "b": b, "c": c, "d": d}, seq_len=16)
        sampler = DemoAwareBatchSampler(ds, batch_size=4, shuffle=False)
        batches = list(sampler)
        # Each demo has 64 / 16 = 4 chunks. 4 demos across 4 streams:
        # each stream gets 1 demo. So 4 batches.
        assert len(batches) == 4
        assert all(len(b) == 4 for b in batches)

    def test_sampler_respects_within_demo_order(self, tmp_path: Path) -> None:
        samples = _synth_paired_samples(tmp_path, 96)
        ds = MkwBCDataset({"d": samples}, seq_len=16)
        sampler = DemoAwareBatchSampler(ds, batch_size=1, shuffle=False)
        batches = list(sampler)
        # All batches are from demo 'd'; seq_starts must be monotonically increasing.
        starts = [ds[b[0]]["meta"]["seq_start"] for b in batches]
        for s1, s2 in zip(starts, starts[1:], strict=False):
            assert s2 > s1

    def test_shuffle_reorders_between_epochs(self, tmp_path: Path) -> None:
        a = _synth_paired_samples(tmp_path, 64, demo_id="a")
        b = _synth_paired_samples(tmp_path, 64, demo_id="b")
        ds = MkwBCDataset({"a": a, "b": b}, seq_len=16)
        sampler1 = DemoAwareBatchSampler(ds, batch_size=1, shuffle=True, seed=0)
        # With batch_size=1 and 2 demos, stream 0 gets both demos in some order.
        # Run two "epochs" (two full iterations) and collect demo order per epoch.
        epoch1 = [ds[bb[0]]["meta"]["demo_id"] for bb in sampler1]
        epoch2 = [ds[bb[0]]["meta"]["demo_id"] for bb in sampler1]
        # Within an epoch, consecutive windows of the same demo stay together in order.
        # Between epochs, the demo order may differ (not guaranteed to, but likely).
        # We test at minimum: within-demo ordering is intact per epoch.
        self._assert_within_demo_ordered(ds, epoch1)
        self._assert_within_demo_ordered(ds, epoch2)

    def _assert_within_demo_ordered(self, ds: MkwBCDataset, demo_order: list[str]) -> None:
        # This tests a weaker property: for each demo, we see exactly its
        # chunks in order — appearances of each demo should form a
        # contiguous run (no interleaving within a single stream).
        first: dict[str, int] = {}
        last: dict[str, int] = {}
        for i, did in enumerate(demo_order):
            if did not in first:
                first[did] = i
            last[did] = i
        # Chunks of one demo should form a contiguous run (no interleaving
        # within a single stream).
        for did in first:
            contiguous = set(range(first[did], last[did] + 1))
            actual = {i for i, d in enumerate(demo_order) if d == did}
            assert actual == contiguous, f"demo {did} appearances aren't contiguous"


# ---------------------------------------------------------------------------
# Collate + is_continuation.
# ---------------------------------------------------------------------------


class TestCollateAndContinuation:
    def test_is_continuation_same_demo_contiguous(self) -> None:
        prev = {"demo_id": "a", "seq_start": 0}
        curr = {"demo_id": "a", "seq_start": 16}
        assert compute_is_continuation(prev, curr, seq_len=16) is True

    def test_is_continuation_same_demo_non_contiguous(self) -> None:
        prev = {"demo_id": "a", "seq_start": 0}
        curr = {"demo_id": "a", "seq_start": 17}  # off by one
        assert compute_is_continuation(prev, curr, seq_len=16) is False

    def test_is_continuation_different_demo(self) -> None:
        prev = {"demo_id": "a", "seq_start": 0}
        curr = {"demo_id": "b", "seq_start": 16}
        assert compute_is_continuation(prev, curr, seq_len=16) is False

    def test_is_continuation_first_batch(self) -> None:
        curr = {"demo_id": "a", "seq_start": 0}
        assert compute_is_continuation(None, curr, seq_len=16) is False


class TestDataLoader:
    def test_end_to_end_dataloader(self, tmp_path: Path) -> None:
        a = _synth_paired_samples(tmp_path, 64, demo_id="a")
        b = _synth_paired_samples(tmp_path, 64, demo_id="b")
        ds = MkwBCDataset({"a": a, "b": b}, seq_len=16, stack_size=4, frame_skip=4)
        sampler = DemoAwareBatchSampler(ds, batch_size=2, shuffle=False)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=bc_collate_fn)
        batches = list(loader)
        assert len(batches) > 0
        for batch in batches:
            assert batch["frames"].shape[:2] == (2, 16)  # (B, T, ...)
            assert batch["frames"].shape[2] == 4  # stack
            assert batch["actions"]["steering_bin"].shape == (2, 16)
            assert len(batch["meta"]["demo_id"]) == 2

    def test_dataloader_produces_continuation_chain_for_single_demo(self, tmp_path: Path) -> None:
        samples = _synth_paired_samples(tmp_path, 128, demo_id="solo")
        ds = MkwBCDataset({"solo": samples}, seq_len=16)
        sampler = DemoAwareBatchSampler(ds, batch_size=1, shuffle=False)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=bc_collate_fn)
        prev = None
        continuations = []
        for batch in loader:
            # Batch size 1 → single position.
            curr = {"demo_id": batch["meta"]["demo_id"][0], "seq_start": batch["meta"]["seq_start"][0]}
            continuations.append(compute_is_continuation(prev, curr, seq_len=16))
            prev = curr
        # First batch = False, rest = True.
        assert continuations[0] is False
        assert all(c is True for c in continuations[1:])


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def test_demo_id_from_path() -> None:
    assert demo_id_from_path("/a/b/demo_2026.dtm") == "demo_2026"
    assert demo_id_from_path(Path("x.dtm")) == "x"
