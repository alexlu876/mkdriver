"""Multi-track BTR environment: gym-compliant wrapper around VIPTankz's Dolphin scripting fork.

See ``docs/TRAINING_METHODOLOGY.md`` for the algorithmic choices (variable
checkpoints, lenient reset, reward structure) and ``docs/PIVOT_2026-04-17.md``
for why we own this code rather than importing VIPTankz's verbatim.

Components:
- ``track_meta`` — loads ``data/track_metadata.yaml`` (WR times → checkpoint counts)
- ``reward`` — per-episode reward accumulator implementing the v2 formula
- ``dolphin_script`` — runs inside Dolphin via ``--script`` (the slave process)
- ``dolphin_env`` — gymnasium-compatible env launched in the training process (master)
"""

from mkw_rl.env.dolphin_env import MkwDolphinEnv, available_tracks
from mkw_rl.env.reward import RaceState, RewardBreakdown, RewardConfig, TrackRewardTracker
from mkw_rl.env.track_meta import TrackMetadata, checkpoint_count_for_track, load_track_metadata

__all__ = [
    "MkwDolphinEnv",
    "RaceState",
    "RewardBreakdown",
    "RewardConfig",
    "TrackMetadata",
    "TrackRewardTracker",
    "available_tracks",
    "checkpoint_count_for_track",
    "load_track_metadata",
]
