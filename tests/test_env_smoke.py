"""Pure-function tests for the env reward + metadata modules.

Does NOT test the gym-env wrapper end-to-end — that requires a live Dolphin
instance and is human-driven. These tests cover everything that can be
checked offline: track metadata parsing, reward component correctness,
reset-threshold firing, and checkpoint scheduling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mkw_rl.env.reward import RaceState, RewardConfig, TrackRewardTracker
from mkw_rl.env.track_meta import (
    TrackMetadata,
    checkpoint_count_for_track,
    load_track_metadata,
)

# ---------------------------------------------------------------------------
# Track metadata.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "track_metadata.yaml"


class TestTrackMetadata:
    def test_default_metadata_loads(self) -> None:
        meta = load_track_metadata()
        # Our shipped YAML has all 32 vanilla tracks.
        assert len(meta) == 32
        assert "luigi_circuit_tt" in meta

    def test_known_track_fields(self) -> None:
        meta = load_track_metadata()
        luigi = meta["luigi_circuit_tt"]
        assert luigi.name == "Luigi Circuit"
        assert luigi.cup == "mushroom"
        assert luigi.laps == 3
        # Luigi WR is 1:08.733 = 68.733s → 100 × 68.733 / 60 ≈ 115 checkpoints/lap.
        assert luigi.n_checkpoints_per_lap == 115
        assert luigi.n_checkpoints_total == 345

    def test_rainbow_road_is_longest(self) -> None:
        """Sanity: Rainbow Road should have the densest checkpoint schedule."""
        meta = load_track_metadata()
        per_lap = {slug: m.n_checkpoints_per_lap for slug, m in meta.items()}
        # N64 Bowser's Castle has the longest WR in our data so it gets the most checkpoints;
        # Rainbow Road is second longest. Sanity check that Rainbow Road is in the top 3.
        top3 = sorted(per_lap.items(), key=lambda x: -x[1])[:3]
        top_slugs = [slug for slug, _ in top3]
        assert "rainbow_road_tt" in top_slugs

    def test_checkpoint_count_for_track_helper(self) -> None:
        assert checkpoint_count_for_track("luigi_circuit_tt") == 115

    def test_unknown_slug_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown track slug"):
            checkpoint_count_for_track("nonexistent_track_tt")

    def test_invalid_yaml_missing_field(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo_tt:\n  name: Foo\n  cup: mushroom\n  wr_seconds: 60\n")  # missing wr_category, laps
        with pytest.raises(ValueError, match="missing required field"):
            load_track_metadata(bad)

    def test_invalid_wr_category(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            yaml.safe_dump(
                {
                    "foo_tt": {
                        "name": "Foo",
                        "cup": "mushroom",
                        "wr_seconds": 60.0,
                        "wr_category": "bogus",
                        "laps": 3,
                    }
                }
            )
        )
        with pytest.raises(ValueError, match="invalid wr_category"):
            load_track_metadata(bad)


# ---------------------------------------------------------------------------
# Reward tracker — hand-constructed track meta + states.
# ---------------------------------------------------------------------------


def _track(
    *,
    wr_seconds: float = 60.0,
    laps: int = 3,
    slug: str = "test_track",
) -> TrackMetadata:
    """Minimal track fixture. wr_seconds=60 gives 100 checkpoints/lap — clean for hand math."""
    return TrackMetadata(
        slug=slug,
        name="Test Track",
        cup="mushroom",
        wr_seconds=wr_seconds,
        wr_category="non_glitch",
        laps=laps,
    )


def _state(
    *,
    race_completion: float = 1.0,
    current_lap: int = 1,
    race_stage: int = 3,
    race_position: int = 1,
    touching_offroad: bool = False,
    wall_collide: int = 0,
    offroad_invincibility: int = 0,
) -> RaceState:
    return RaceState(
        race_completion=race_completion,
        current_lap=current_lap,
        race_stage=race_stage,
        race_position=race_position,
        touching_offroad=touching_offroad,
        wall_collide=wall_collide,
        offroad_invincibility=offroad_invincibility,
    )


class TestCheckpointSchedule:
    def test_thresholds_uniform(self) -> None:
        """100 checkpoints/lap × 3 laps = 300 thresholds from just-past-1.0 to 4.0."""
        t = TrackRewardTracker(_track(wr_seconds=60.0, laps=3))
        # 100 per lap × 3 laps + sentinel.
        assert len(t.checkpoints) == 301
        # First threshold: 1.0 + 1/100 = 1.01.
        assert t.checkpoints[0] == pytest.approx(1.01)
        # Last non-sentinel: 4.0 exactly.
        assert t.checkpoints[299] == pytest.approx(4.0)
        # Sentinel.
        assert t.checkpoints[-1] == 9999.0

    def test_variable_density_by_wr(self) -> None:
        short = TrackRewardTracker(_track(wr_seconds=30.0))  # 50 checkpoints/lap
        long_ = TrackRewardTracker(_track(wr_seconds=150.0))  # 250 checkpoints/lap
        # Non-sentinel thresholds count.
        assert len(short.checkpoints) - 1 == 50 * 3
        assert len(long_.checkpoints) - 1 == 250 * 3

    def test_expected_frames_per_checkpoint(self) -> None:
        t = TrackRewardTracker(_track(wr_seconds=60.0, laps=3))
        # 60s × 50fps / (100 × 3) = 3000 / 300 = 10 frames/checkpoint at WR pace.
        assert t.expected_frames_per_checkpoint == 10


class TestCheckpointReward:
    def test_no_reward_before_first_threshold(self) -> None:
        t = TrackRewardTracker(_track())
        rb, done = t.step(_state(race_completion=1.005))  # hasn't crossed 1.01 yet
        assert rb.checkpoint == 0.0
        assert not done

    def test_reward_fires_at_threshold_cross(self) -> None:
        t = TrackRewardTracker(_track())
        rb, done = t.step(_state(race_completion=1.015))  # crossed 1.01
        assert rb.checkpoint > 0.0
        assert not done

    def test_reward_fires_once_per_checkpoint(self) -> None:
        t = TrackRewardTracker(_track())
        # Step past the first checkpoint.
        t.step(_state(race_completion=1.015))
        # Step again at same position — no additional reward.
        rb, _ = t.step(_state(race_completion=1.015))
        assert rb.checkpoint == 0.0

    def test_multiple_checkpoints_in_one_frame(self) -> None:
        """At very high speed, we can cross multiple thresholds between frames."""
        t = TrackRewardTracker(_track())
        initial_idx = t.current_checkpoint
        rb, _ = t.step(_state(race_completion=1.035))  # crosses 1.01, 1.02, 1.03
        # Three checkpoints fired and reward is positive (exact value depends
        # on per_hit_base which is normalized by n_checkpoints_per_lap).
        assert t.current_checkpoint == initial_idx + 3
        assert rb.checkpoint >= 3 * t.per_hit_base
        assert rb.checkpoint <= 3 * t.per_hit_base * t.config.speed_bonus_max

    def test_speed_bonus_higher_for_faster_hits(self) -> None:
        # Fast hit: one frame of gap. Slow hit: expected_frames_per_checkpoint
        # frames of gap.
        t_fast = TrackRewardTracker(_track())
        t_slow = TrackRewardTracker(_track())

        # Fast: crosses first checkpoint after 1 frame.
        rb_fast, _ = t_fast.step(_state(race_completion=1.015))

        # Slow: sit on 1.005 for many frames, then cross.
        for _ in range(t_slow.expected_frames_per_checkpoint):
            t_slow.step(_state(race_completion=1.005))
        rb_slow, _ = t_slow.step(_state(race_completion=1.015))

        assert rb_fast.checkpoint > rb_slow.checkpoint


class TestPenalties:
    def test_offroad_penalty_applied(self) -> None:
        t = TrackRewardTracker(_track())
        rb, _ = t.step(_state(race_completion=1.0, touching_offroad=True))
        assert rb.offroad == -RewardConfig().offroad_penalty

    def test_offroad_invincibility_exempts_penalty(self) -> None:
        t = TrackRewardTracker(_track())
        rb, _ = t.step(
            _state(race_completion=1.0, touching_offroad=True, offroad_invincibility=90)
        )
        assert rb.offroad == 0.0

    def test_wall_penalty_applied(self) -> None:
        t = TrackRewardTracker(_track())
        rb, _ = t.step(_state(race_completion=1.0, wall_collide=1))
        assert rb.wall == -RewardConfig().wall_penalty

    def test_penalty_is_larger_for_wall_than_offroad(self) -> None:
        """Sanity check: wall contact is worse than driving off-road."""
        cfg = RewardConfig()
        assert cfg.wall_penalty > cfg.offroad_penalty


class TestFinishBonus:
    def test_finish_fires_once(self) -> None:
        t = TrackRewardTracker(_track())
        rb1, done1 = t.step(_state(race_completion=4.0, race_position=1))
        assert rb1.finish == 10.0
        assert rb1.position == 12 * 0.5  # (13 - 1) × 0.5
        assert done1

        rb2, _ = t.step(_state(race_completion=4.0, race_position=1))
        assert rb2.finish == 0.0
        assert rb2.position == 0.0

    def test_position_bonus_scales_inversely(self) -> None:
        t1 = TrackRewardTracker(_track())
        t12 = TrackRewardTracker(_track())
        rb1, _ = t1.step(_state(race_completion=4.0, race_position=1))
        rb12, _ = t12.step(_state(race_completion=4.0, race_position=12))
        assert rb1.position > rb12.position


class TestResetThreshold:
    def test_reset_fires_after_idle(self) -> None:
        t = TrackRewardTracker(_track(), RewardConfig(reset_threshold_frames=50))
        for _ in range(50):
            _, done = t.step(_state(race_completion=1.0))
            assert not done
        _, done = t.step(_state(race_completion=1.0))
        assert done

    def test_reset_counter_resets_on_checkpoint(self) -> None:
        t = TrackRewardTracker(_track(), RewardConfig(reset_threshold_frames=50))
        # 30 frames no progress.
        for _ in range(30):
            t.step(_state(race_completion=1.0))
        # Cross a checkpoint.
        t.step(_state(race_completion=1.015))
        # Should be able to go ~50 more frames.
        for _ in range(45):
            _, done = t.step(_state(race_completion=1.015))
            assert not done


class TestAlignToState:
    def test_align_to_mid_race_savestate(self) -> None:
        """Simulates loading a savestate at race_completion=1.7 (lap 1 ~70%)."""
        t = TrackRewardTracker(_track())  # 100 checkpoints/lap
        t.align_to_state(_state(race_completion=1.7))
        # Should have skipped past ~69 checkpoints already (1.01, 1.02, ..., 1.69).
        # First un-crossed threshold is 1.70 (or 1.71 depending on rounding).
        assert t.current_checkpoint >= 69

    def test_align_emits_no_reward(self) -> None:
        """align_to_state must not retroactively fire rewards."""
        t = TrackRewardTracker(_track())
        t.align_to_state(_state(race_completion=1.7))
        # Step at the same race_completion — no new rewards.
        rb, _ = t.step(_state(race_completion=1.7))
        assert rb.checkpoint == 0.0


class TestRewardBreakdown:
    def test_total_is_sum(self) -> None:
        t = TrackRewardTracker(_track())
        rb, _ = t.step(
            _state(race_completion=1.015, touching_offroad=True, wall_collide=1)
        )
        assert rb.total == pytest.approx(rb.checkpoint + rb.offroad + rb.wall)

    def test_as_dict_has_expected_keys(self) -> None:
        rb, _ = TrackRewardTracker(_track()).step(_state())
        d = rb.as_dict()
        assert set(d) == {"checkpoint", "offroad", "wall", "finish", "position", "total"}
