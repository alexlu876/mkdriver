"""Tests for src/mkw_rl/dtm/action_encoding.py."""

from __future__ import annotations

import random

import pytest

from mkw_rl.dtm.action_encoding import (
    N_STEERING_BINS,
    bin_width,
    center_bin,
    decode_steering,
    encode_steering,
)


class TestBasicEncoding:
    def test_n_bins_is_21_and_odd(self) -> None:
        assert N_STEERING_BINS == 21
        assert N_STEERING_BINS % 2 == 1

    def test_zero_is_center_bin(self) -> None:
        assert encode_steering(0.0) == center_bin() == 10

    def test_minus_one_is_first_bin(self) -> None:
        assert encode_steering(-1.0) == 0

    def test_plus_one_is_last_bin(self) -> None:
        assert encode_steering(1.0) == N_STEERING_BINS - 1 == 20

    def test_out_of_range_clipped(self) -> None:
        assert encode_steering(-2.0) == 0
        assert encode_steering(2.0) == N_STEERING_BINS - 1
        assert encode_steering(float("-inf")) == 0
        assert encode_steering(float("inf")) == N_STEERING_BINS - 1


class TestDecoding:
    def test_center_decodes_to_zero(self) -> None:
        assert decode_steering(center_bin()) == pytest.approx(0.0, abs=1e-9)

    def test_first_bin_is_near_minus_one(self) -> None:
        # Bin 0 center is -1 + 0.5 * width = -1 + width/2.
        assert decode_steering(0) == pytest.approx(-1.0 + bin_width() / 2, abs=1e-9)

    def test_last_bin_is_near_plus_one(self) -> None:
        assert decode_steering(N_STEERING_BINS - 1) == pytest.approx(1.0 - bin_width() / 2, abs=1e-9)

    def test_monotonic(self) -> None:
        vals = [decode_steering(i) for i in range(N_STEERING_BINS)]
        for a, b in zip(vals, vals[1:], strict=False):
            assert a < b

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_steering(-1)
        with pytest.raises(ValueError):
            decode_steering(N_STEERING_BINS)


class TestRoundTrip:
    def test_exact_centers_round_trip(self) -> None:
        # decode → encode should return the original bin for all bin centers.
        for i in range(N_STEERING_BINS):
            center = decode_steering(i)
            assert encode_steering(center) == i

    def test_random_values_within_half_bin(self) -> None:
        """encode(decode(encode(x))) == encode(x); |x - decode(encode(x))| <= bin_width/2."""
        rng = random.Random(0)
        for _ in range(1000):
            x = rng.uniform(-1.0, 1.0)
            encoded = encode_steering(x)
            decoded = decode_steering(encoded)
            assert abs(x - decoded) <= bin_width() / 2 + 1e-9
            # Fixed point after one pass.
            assert encode_steering(decoded) == encoded


class TestBoundaryBehavior:
    def test_boundary_near_bin_edge(self) -> None:
        # The edge between bin 0 and bin 1 is at -1 + width. Values just
        # above that edge go to bin 1; just below go to bin 0.
        edge = -1.0 + bin_width()
        assert encode_steering(edge - 1e-6) == 0
        # Right at the edge (0 + width_offset) should go to bin 1.
        assert encode_steering(edge + 1e-6) == 1

    def test_plus_one_inclusive(self) -> None:
        """+1.0 exactly maps to the last bin, not past it."""
        assert encode_steering(1.0) == N_STEERING_BINS - 1
        # Just barely inside +1 — still last bin.
        assert encode_steering(1.0 - 1e-9) == N_STEERING_BINS - 1


class TestBinWidth:
    def test_bin_width_is_consistent(self) -> None:
        assert bin_width() == pytest.approx(2.0 / N_STEERING_BINS)
