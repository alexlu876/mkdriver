"""Tests for ``src/mkw_rl/rl/track_sampler.py``.

Pure-function tests — no env, no Dolphin, no torch. Covers:
- construction invariants (non-empty, no duplicates)
- cold-start uniformity
- progress EMA updates
- weight formula (max - p + ε)
- inverse-progress sampling bias (low-progress → higher sampling rate)
- reset() wipes state
- add_track() appends without resetting existing
- determinism with seeded RNG
- single-track edge case
- unknown-slug update raises
"""

from __future__ import annotations

import collections

import pytest

from mkw_rl.rl.track_sampler import (
    ProgressWeightedTrackSampler,
    TrackSamplerConfig,
    construct_from_available,
)

# ---------------------------------------------------------------------------
# Construction.
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_track_list_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ProgressWeightedTrackSampler(track_slugs=[])

    def test_duplicate_slugs_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ProgressWeightedTrackSampler(
                track_slugs=["luigi", "luigi", "rainbow"]
            )

    def test_single_track(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["luigi"])
        assert s.n_tracks == 1
        # Only choice — always returns luigi.
        for _ in range(5):
            assert s.sample() == "luigi"

    def test_initial_progress_all_zero(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c"])
        assert s.progress == {"a": 0.0, "b": 0.0, "c": 0.0}

    def test_construct_from_available(self) -> None:
        s = construct_from_available(["luigi_circuit_tt", "rainbow_road_tt"])
        assert s.n_tracks == 2
        assert set(s.progress.keys()) == {"luigi_circuit_tt", "rainbow_road_tt"}


# ---------------------------------------------------------------------------
# Cold-start uniformity.
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_cold_start_weights_uniform(self) -> None:
        """All zeros → weights all equal to epsilon → uniform probabilities."""
        s = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c", "d"])
        weights = s.weights()
        assert weights == {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1}
        dist = s.distribution()
        for slug, p in dist.items():
            assert p == pytest.approx(0.25), f"{slug} got {p}, expected 0.25"

    def test_cold_start_sampling_roughly_uniform(self) -> None:
        """With equal progress, many draws should approximate uniform distribution."""
        s = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c", "d"], seed=0)
        counts = collections.Counter(s.sample() for _ in range(4000))
        for slug in ["a", "b", "c", "d"]:
            # Expect ~1000 ± 5% (large enough N that chi-square would accept).
            assert 900 <= counts[slug] <= 1100, f"{slug}: {counts[slug]}"


# ---------------------------------------------------------------------------
# EMA updates.
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_first_update_is_alpha_fraction_of_return(self) -> None:
        """With α=0.05 and initial progress=0, first update of 10.0 gives 0.5."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["a", "b"], config=TrackSamplerConfig(ema_alpha=0.05)
        )
        s.update("a", 10.0)
        assert s.progress["a"] == pytest.approx(0.5)
        assert s.progress["b"] == 0.0

    def test_updates_converge_to_return(self) -> None:
        """Repeated identical updates converge to the return value."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["a"], config=TrackSamplerConfig(ema_alpha=0.1)
        )
        for _ in range(200):
            s.update("a", 5.0)
        # After 200 iterations with α=0.1, we're ~1-0.9^200 ≈ ~100% converged.
        assert s.progress["a"] == pytest.approx(5.0, abs=1e-6)

    def test_unknown_slug_raises(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["a", "b"])
        with pytest.raises(KeyError, match="unknown track slug"):
            s.update("ghost_track", 5.0)

    def test_update_preserves_other_tracks(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c"])
        s.update("b", 7.5)
        assert s.progress["a"] == 0.0
        assert s.progress["c"] == 0.0
        assert s.progress["b"] != 0.0


# ---------------------------------------------------------------------------
# Weight formula + inverse-progress bias.
# ---------------------------------------------------------------------------


class TestWeightFormula:
    def test_leader_gets_epsilon_weight(self) -> None:
        """The track with the highest progress has weight exactly epsilon."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["easy", "hard"],
            config=TrackSamplerConfig(epsilon=0.1, ema_alpha=1.0),
        )
        s.update("easy", 10.0)  # progress=10 (α=1 → no smoothing)
        s.update("hard", 2.0)  # progress=2
        w = s.weights()
        assert w["easy"] == pytest.approx(0.1)  # max - max + ε = ε
        assert w["hard"] == pytest.approx(10.0 - 2.0 + 0.1)

    def test_low_progress_track_sampled_more(self) -> None:
        """The track with lowest progress should be sampled disproportionately."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["easy", "hard"],
            config=TrackSamplerConfig(epsilon=0.1, ema_alpha=1.0),
            seed=0,
        )
        s.update("easy", 10.0)
        s.update("hard", 2.0)
        counts = collections.Counter(s.sample() for _ in range(4000))
        # Expected ratio: easy has weight 0.1, hard has 8.1 → hard gets
        # 0.1/8.2 ≈ 1.2% easy, 98.8% hard. Allow wide tolerance.
        assert counts["hard"] > counts["easy"] * 30

    def test_three_tracks_ordering(self) -> None:
        """With progresses 0, 5, 10: weights should be 10+ε, 5+ε, ε."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["slow", "med", "fast"],
            config=TrackSamplerConfig(epsilon=0.1, ema_alpha=1.0),
        )
        s.update("slow", 0.0)
        s.update("med", 5.0)
        s.update("fast", 10.0)
        w = s.weights()
        assert w["slow"] == pytest.approx(10.1)
        assert w["med"] == pytest.approx(5.1)
        assert w["fast"] == pytest.approx(0.1)

    def test_solved_track_weight_floor(self) -> None:
        """As a track's progress approaches the leader, its weight approaches ε."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["a", "b"],
            config=TrackSamplerConfig(epsilon=0.1, ema_alpha=1.0),
        )
        s.update("a", 10.0)
        s.update("b", 9.95)
        w = s.weights()
        assert w["a"] == pytest.approx(0.1)  # leader → epsilon
        assert w["b"] == pytest.approx(0.15)  # close follower → slightly above ε


# ---------------------------------------------------------------------------
# Distribution (normalized) matches weights.
# ---------------------------------------------------------------------------


class TestDistribution:
    def test_distribution_sums_to_one(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c", "d"])
        s.update("a", 3.0)
        s.update("c", 7.0)
        total = sum(s.distribution().values())
        assert total == pytest.approx(1.0)

    def test_distribution_has_all_tracks(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["x", "y", "z"])
        dist = s.distribution()
        assert set(dist.keys()) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_sequence(self) -> None:
        s1 = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c"], seed=42)
        s2 = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c"], seed=42)
        seq1 = [s1.sample() for _ in range(50)]
        seq2 = [s2.sample() for _ in range(50)]
        assert seq1 == seq2

    def test_different_seeds_different_sequences(self) -> None:
        """Weak test — two random seeds should produce different draws
        at least once in 50 samples. Probability of collision is ~(1/3)^50."""
        s1 = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c"], seed=1)
        s2 = ProgressWeightedTrackSampler(track_slugs=["a", "b", "c"], seed=2)
        seq1 = [s1.sample() for _ in range(50)]
        seq2 = [s2.sample() for _ in range(50)]
        assert seq1 != seq2


# ---------------------------------------------------------------------------
# reset / add_track.
# ---------------------------------------------------------------------------


class TestResetAndAdd:
    def test_reset_wipes_progress(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["a", "b"])
        s.update("a", 10.0)
        s.update("b", 5.0)
        assert s.progress["a"] != 0.0
        s.reset()
        assert s.progress == {"a": 0.0, "b": 0.0}

    def test_add_track_inserts_at_cold_start(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["a", "b"])
        s.update("a", 10.0)  # some progress
        s.add_track("c")
        assert s.n_tracks == 3
        assert s.progress["c"] == 0.0
        # a's progress is unchanged.
        assert s.progress["a"] == pytest.approx(0.5)  # α=0.05 × 10

    def test_add_existing_track_raises(self) -> None:
        s = ProgressWeightedTrackSampler(track_slugs=["a"])
        with pytest.raises(KeyError, match="already present"):
            s.add_track("a")

    def test_added_track_gets_high_initial_weight(self) -> None:
        """A newly-added track should be one of the most-sampled until it
        accumulates progress, because its weight = max - 0 + ε is near-max."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["a"],
            config=TrackSamplerConfig(epsilon=0.1, ema_alpha=1.0),
        )
        s.update("a", 10.0)  # make a very "solved"
        s.add_track("newcomer")
        w = s.weights()
        assert w["a"] == pytest.approx(0.1)
        assert w["newcomer"] == pytest.approx(10.1)


# ---------------------------------------------------------------------------
# End-to-end scenario: curriculum behavior over many episodes.
# ---------------------------------------------------------------------------


class TestCurriculumBehavior:
    def test_eventually_balances_out(self) -> None:
        """Simulation: agent makes uniform progress across all tracks. After
        many episodes the sampling distribution should tend toward uniform
        (because all progress values tend toward the same value)."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["a", "b", "c"],
            config=TrackSamplerConfig(ema_alpha=0.1, epsilon=0.1),
            seed=7,
        )
        for _ in range(500):
            slug = s.sample()
            # Every track returns the same reward — mock "equally easy" tracks.
            s.update(slug, 5.0)

        dist = s.distribution()
        for slug in ["a", "b", "c"]:
            # Each should be close to 1/3.
            assert 0.25 < dist[slug] < 0.42, f"{slug} prob {dist[slug]}"

    def test_hard_track_gets_oversampled(self) -> None:
        """Simulation: 'hard' track gives low reward, 'easy' gives high. The
        sampler should push more episodes onto 'hard' over time."""
        s = ProgressWeightedTrackSampler(
            track_slugs=["easy", "hard"],
            config=TrackSamplerConfig(ema_alpha=0.1, epsilon=0.1),
            seed=7,
        )
        rewards = {"easy": 10.0, "hard": 1.0}
        sample_counts: dict[str, int] = collections.Counter()
        for _ in range(1000):
            slug = s.sample()
            sample_counts[slug] += 1
            s.update(slug, rewards[slug])

        # Hard should be sampled more than easy over the full run.
        assert sample_counts["hard"] > sample_counts["easy"]
