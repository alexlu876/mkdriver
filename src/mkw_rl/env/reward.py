"""Per-episode reward accumulator implementing the v2 methodology.

See ``docs/TRAINING_METHODOLOGY.md`` §5 for the reward structure and
``DolphinScript.py`` in VIPTankz's repo for the underlying v1 reference
that we're extending.

v2 changes layered on top of VIPTankz's published v1:
- Variable checkpoints per track (``n_checkpoints_per_lap`` from WR time)
- Off-road penalty per frame (v1 has none)
- Wall penalty per frame (v1 has none)
- Lenient reset threshold (configurable; default matches v2 video's
  "1 second of progress per 15 seconds" rule-of-thumb)
- Checkpoint × speed-bonus multiplier (v1 fires a flat +1 per checkpoint)

Pure Python — intentionally no numpy/torch/yaml deps so this module is
importable from inside Dolphin's embedded interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mkw_rl.env.track_meta import TrackMetadata


@dataclass(frozen=True)
class RewardConfig:
    """All tunable constants. Defaults picked from v2 methodology + VIPTankz.

    Tune ``offroad_penalty`` and ``wall_penalty`` carefully: too high and the
    agent learns to drive backwards to avoid walls; too low and they have no
    effect. See the "backward-driving trap" note in TRAINING_METHODOLOGY.md.
    """

    # Checkpoint reward. Per ``docs/TRAINING_METHODOLOGY.md`` §5, the per-hit
    # base is normalized by checkpoint count so cumulative per-lap checkpoint
    # reward is ~constant across tracks regardless of WR time. The tracker
    # computes actual_per_hit_base = checkpoint_reward_per_lap / n_checkpoints.
    # speed_bonus multiplies that by [1, speed_bonus_max] depending on how
    # fast the checkpoint was hit vs WR pace.
    checkpoint_reward_per_lap: float = 1.0
    speed_bonus_alpha: float = 0.5  # linear scale factor
    speed_bonus_max: float = 2.0  # clamp so a single fast checkpoint can't dominate

    # Per-frame penalties (new in v2; zero in v1).
    offroad_penalty: float = 0.01
    wall_penalty: float = 0.05

    # Terminal rewards. Kept in the same scale as cumulative checkpoint
    # reward (~1 per lap × laps × speed_bonus ≈ 3-6): a 10.0 finish bonus
    # roughly doubles the total race reward, making finishing strictly
    # preferred over timing out via reset threshold.
    finish_bonus: float = 10.0
    position_bonus_scale: float = 0.5  # (13 - pos) × scale; 13 matches VIPTankz's formula

    # Lenient reset threshold, expressed in frames since the last checkpoint
    # hit. At PAL 50 fps, 750 frames = 15 seconds — matches v2's "1s progress
    # per 15s" spirit since checkpoints are dense enough to cross one per
    # ~0.3-1s at WR pace across all tracks.
    reset_threshold_frames: int = 750

    # Emulation framerate. Used to compute expected frames-per-checkpoint at
    # WR pace. PAL MKWii is 50 Hz.
    emulation_fps: int = 50


@dataclass
class RaceState:
    """Subset of RAM-read values the reward function consumes.

    Populated by ``dolphin_script.py``'s memory-tracker on each frame.
    Ordering / naming matches VIPTankz's ``DolphinScript.py`` vars so
    porting is mechanical.
    """

    race_completion: float  # 1.0 → 4.0 over 3 laps per VIPTankz
    current_lap: int
    # Stored as u32 in game memory (must be read via memory.read_u32 — see
    # dolphin_script.py:_read_race_state). VIPTankz only ever branches on
    # ``stage == 4`` meaning "race ended"; other values (0, 1, 2, 3) are
    # inferred from community RE to correspond to pre-race states +
    # active racing, but VIPTankz's code doesn't rely on them and neither
    # do we — our reward function uses ``race_completion`` for progression
    # and ``race_completion >= 4.0`` for finish detection.
    race_stage: int
    race_position: int  # 1-12 finishing order
    touching_offroad: bool
    wall_collide: int  # non-zero = touching wall
    offroad_invincibility: int  # >0 = mushroom/boost protection from offroad


@dataclass
class RewardBreakdown:
    """Component-by-component reward log, emitted per step for wandb logging."""

    checkpoint: float = 0.0
    offroad: float = 0.0
    wall: float = 0.0
    finish: float = 0.0
    position: float = 0.0

    @property
    def total(self) -> float:
        return self.checkpoint + self.offroad + self.wall + self.finish + self.position

    def as_dict(self) -> dict[str, float]:
        return {
            "checkpoint": self.checkpoint,
            "offroad": self.offroad,
            "wall": self.wall,
            "finish": self.finish,
            "position": self.position,
            "total": self.total,
        }


@dataclass
class TrackRewardTracker:
    """Per-episode reward state machine.

    Constructed at ``env.reset()``; ``step(state)`` called on every frame.
    Checkpoint thresholds are uniform divisions of ``RaceCompletion`` from
    1.0 (lap 1 start) to 4.0 (3 laps done), matching VIPTankz's scheme
    (``DolphinScript.py:427-445``) but with our v2 variable-density count.
    """

    track_meta: TrackMetadata
    config: RewardConfig = field(default_factory=RewardConfig)

    checkpoints: list[float] = field(init=False)
    current_checkpoint: int = field(init=False, default=0)
    frames_since_checkpoint: int = field(init=False, default=0)
    expected_frames_per_checkpoint: int = field(init=False)
    # Per-hit base reward, derived as config.checkpoint_reward_per_lap / n_per_lap.
    # Populated in __post_init__.
    per_hit_base: float = field(init=False)
    _finished: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        n_per_lap = self.track_meta.n_checkpoints_per_lap
        laps = self.track_meta.laps

        # VIPTankz's layout: start=1.0, end=4.0, lap_length=1.0 per lap.
        # We keep the same RaceCompletion scheme; only the density changes.
        self.checkpoints = []
        for lap in range(laps):
            lap_start = 1.0 + lap * 1.0
            step = 1.0 / n_per_lap
            for i in range(1, n_per_lap + 1):
                self.checkpoints.append(lap_start + i * step)
        # Sentinel so indexing by `current_checkpoint` never goes OOB post-finish.
        self.checkpoints.append(9999.0)

        # Per-hit base, normalized so per-lap cumulative checkpoint reward
        # equals config.checkpoint_reward_per_lap (before speed bonus).
        # This prevents long tracks from dominating gradient signal.
        self.per_hit_base = self.config.checkpoint_reward_per_lap / n_per_lap

        # Expected frames per checkpoint at WR pace. Used for speed_bonus scaling.
        # WR time covers the full race (3 laps). race_completion delta per WR = 3.0.
        # Frames per WR = wr_seconds × fps. Checkpoints per WR = n_per_lap × laps.
        # So expected frames/checkpoint = (wr_seconds × fps) / (n_per_lap × laps).
        total_checkpoints = n_per_lap * laps
        self.expected_frames_per_checkpoint = max(
            1, int(self.track_meta.wr_seconds * self.config.emulation_fps / total_checkpoints)
        )

    def align_to_state(self, state: RaceState) -> None:
        """Advance ``current_checkpoint`` past any already-crossed thresholds.

        Needed when the savestate starts mid-race (VIPTankz's default) — otherwise
        our first step() would fire dozens of retroactive checkpoint rewards.
        """
        while state.race_completion > self.checkpoints[self.current_checkpoint]:
            self.current_checkpoint += 1

    def step(self, state: RaceState) -> tuple[RewardBreakdown, bool, bool]:
        """Advance one frame. Returns (reward breakdown, terminated, truncated).

        terminated: MDP-end condition (finish line crossed) — a true episode end.
        truncated: time-limit style end (lenient reset fired) — episode cut short
            by the trainer rather than by the MDP. Gymnasium draws this distinction
            explicitly; Q-learning targets differ between the two (bootstrap on
            truncated, don't on terminated).
        """
        breakdown = RewardBreakdown()
        terminated = False
        truncated = False

        self.frames_since_checkpoint += 1

        # Checkpoint crossing — may cross multiple in a single frame at high speed.
        while state.race_completion > self.checkpoints[self.current_checkpoint]:
            # Speed bonus: 1.0 → speed_bonus_max as elapsed → 0.
            elapsed = self.frames_since_checkpoint
            raw_bonus = 1.0 + self.config.speed_bonus_alpha * max(
                0.0, 1.0 - elapsed / self.expected_frames_per_checkpoint
            )
            speed_bonus = min(raw_bonus, self.config.speed_bonus_max)
            breakdown.checkpoint += self.per_hit_base * speed_bonus
            self.current_checkpoint += 1
            self.frames_since_checkpoint = 0  # reset for the NEXT checkpoint's timing

        # Per-frame penalties. Off-road invincibility (mushroom/boost) exempts.
        if state.touching_offroad and state.offroad_invincibility == 0:
            breakdown.offroad = -self.config.offroad_penalty
        if state.wall_collide:
            breakdown.wall = -self.config.wall_penalty

        # Finish — emits finish + position bonuses once, then no further updates.
        if not self._finished and state.race_completion >= 4.0:
            breakdown.finish = self.config.finish_bonus
            breakdown.position = (13 - state.race_position) * self.config.position_bonus_scale
            self._finished = True
            terminated = True

        # Lenient reset — no progress for too long.
        if not terminated and self.frames_since_checkpoint > self.config.reset_threshold_frames:
            truncated = True

        return breakdown, terminated, truncated
