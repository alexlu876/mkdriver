"""Progress-weighted track sampler for multi-track BTR curriculum.

Per ``docs/TRAINING_METHODOLOGY.md`` §4: the env is curriculum-dumb, so this
module owns the per-track weighting. The training loop calls
``sampler.sample()`` before each ``env.reset(track_slug=...)`` and
``sampler.update(slug, episode_return)`` after each episode completes.

Algorithm
---------
Each track carries an EMA of its per-episode return. After every episode::

    progress[slug] ← (1 - α) · progress[slug]  +  α · episode_return

The sampler weights each track by how far behind the leader it is, plus a
small epsilon floor so every track keeps nonzero probability::

    weight[slug] = max(progress) - progress[slug] + epsilon

Normalized to a probability distribution, this biases sampling toward tracks
where the agent is making the least progress — the "failed progression"
cases the v2 video (aLw43abG-NA) explicitly called out. As a track gets
"solved" (progress approaches the max) its weight collapses to just
epsilon — effectively a baseline sampling floor — and the sampler
redistributes to harder tracks without manual intervention.

Cold start: all tracks initialize at ``progress=0``, giving uniform
weights (every weight = 0 + ε = ε). See the "Per-track curriculum seed
weights" note in the methodology doc for a possible WR-difficulty-based
warm start, deferred until uniform cold-start is shown to be a problem.

This module is **pure** — no torch / numpy tensor state, no I/O. The
training loop is responsible for wandb logging of the returned
``distribution()`` dict.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackSamplerConfig:
    """Tunable parameters for the progress-weighted sampler.

    Defaults chosen for the v2 methodology's "1-2 day training run on a
    4090" regime; revisit if episodes are much shorter/longer than
    expected (~1-2 min each).
    """

    # EMA smoothing factor for per-track progress. α=0.05 gives the EMA a
    # half-life of ~14 episodes (ln 0.5 / ln 0.95 ≈ 13.5) — long enough to
    # smooth noise, short enough to respond to policy improvement within
    # a training session. v2 video doesn't publish a value; this is a
    # reasonable default.
    ema_alpha: float = 0.05

    # Epsilon floor: additive smoothing so even a "solved" track (weight
    # ≈ 0 pre-eps) still gets a baseline sampling probability. Larger ε
    # → more uniform sampling; smaller ε → sharper curriculum. 0.1 on
    # a reward scale of ~10 (our finish bonus) gives ~1% floor probability
    # on solved tracks when the leader is ~10 reward ahead.
    epsilon: float = 0.1

    # Initial per-track progress for tracks added with no history. Zero
    # gives uniform cold-start weights. If the leader progress is ever
    # expected to be negative (reward shaping weirdness), consider
    # setting this to something lower so new tracks are explicitly
    # "hardest" until they prove otherwise.
    cold_start_progress: float = 0.0


@dataclass
class ProgressWeightedTrackSampler:
    """Pick the next track to train on, weighted by inverse-progress EMA.

    Usage
    -----
    ::

        sampler = ProgressWeightedTrackSampler(
            track_slugs=["luigi_circuit_tt", "mushroom_gorge_tt", ...],
            seed=42,
        )

        while training:
            slug = sampler.sample()
            obs, info = env.reset(options={"track_slug": slug})
            ... rollout ...
            sampler.update(slug, episode_return)
            # Optional wandb: log sampler.distribution() — maps slug → prob

    Seed semantics
    --------------
    ``seed=None`` (default) uses numpy's entropy-based seed initializer —
    two samplers constructed without a seed produce different sequences.
    Pass an explicit integer for reproducible training runs.

    Thread safety
    -------------
    **Single-writer only.** ``update()`` and ``add_track()`` mutate
    ``self.progress``; ``sample()`` iterates over it. If these are called
    from different threads without external locking, expect ``RuntimeError:
    dictionary changed size during iteration``. For single-threaded
    training loops (the intended use), this is not an issue.
    """

    track_slugs: list[str]
    config: TrackSamplerConfig = field(default_factory=TrackSamplerConfig)
    seed: int | None = None

    progress: dict[str, float] = field(init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not self.track_slugs:
            raise ValueError("track_slugs must be non-empty")
        # dict preserves insertion order (Py 3.7+), which makes sample()
        # deterministic under a seeded RNG.
        self.progress = {
            slug: self.config.cold_start_progress for slug in self.track_slugs
        }
        # Detect duplicate slugs — silently deduping would confuse wandb
        # logging (multiple entries for the same track) and is almost
        # certainly a caller bug.
        if len(self.progress) != len(self.track_slugs):
            dupes = [s for s in self.track_slugs if self.track_slugs.count(s) > 1]
            raise ValueError(f"duplicate track slugs not allowed: {sorted(set(dupes))}")
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # Read / sample.
    # ------------------------------------------------------------------

    def weights(self) -> dict[str, float]:
        """Current un-normalized sampling weights per track.

        ``weight[slug] = max(progress) - progress[slug] + epsilon``. Always
        ≥ ε for every track (strictly positive), so normalization never
        divides by zero.
        """
        max_progress = max(self.progress.values())
        eps = self.config.epsilon
        return {slug: max_progress - p + eps for slug, p in self.progress.items()}

    def distribution(self) -> dict[str, float]:
        """Current normalized sampling probabilities per track — for logging.

        Returns a ``{slug: probability}`` dict that sums to 1. The training
        loop can feed this to wandb as ``track_sampler/{slug}/weight``.
        """
        w = self.weights()
        total = sum(w.values())
        return {slug: val / total for slug, val in w.items()}

    def sample(self) -> str:
        """Draw the next track slug to train on."""
        slugs = list(self.progress.keys())
        weights = self.weights()
        probs = np.array([weights[s] for s in slugs], dtype=np.float64)
        probs /= probs.sum()
        return str(self._rng.choice(slugs, p=probs))

    # ------------------------------------------------------------------
    # Write.
    # ------------------------------------------------------------------

    def update(self, track_slug: str, episode_return: float) -> None:
        """Record an episode's outcome. Updates the EMA for ``track_slug``.

        Must be called after each episode so the sampler's weights reflect
        the policy's current progress on each track.
        """
        if track_slug not in self.progress:
            raise KeyError(
                f"unknown track slug {track_slug!r}; sampler was constructed "
                f"with {sorted(self.progress)}"
            )
        alpha = self.config.ema_alpha
        prev = self.progress[track_slug]
        self.progress[track_slug] = (1.0 - alpha) * prev + alpha * float(episode_return)

    def reset(self) -> None:
        """Wipe all per-track progress back to ``cold_start_progress``.

        Use when the policy is replaced (e.g., loading a checkpoint that's
        ahead of the current sampler state) or when resuming training after
        a long break where old EMAs are no longer meaningful.

        Does NOT reset the RNG — reproducibility under seed is a separate
        concern from curriculum state, and most callers who reset also want
        the RNG to keep advancing so subsequent ``sample()`` calls differ
        from the ones that happened pre-reset.
        """
        for slug in self.progress:
            self.progress[slug] = self.config.cold_start_progress

    # ------------------------------------------------------------------
    # Introspection helpers — for tests + debugging.
    # ------------------------------------------------------------------

    def add_track(self, slug: str) -> None:
        """Add a newly-available track (e.g., user just recorded a savestate).

        Inserts at ``cold_start_progress`` so it gets a near-max weight on
        the next ``sample()``. If the slug already exists, raises KeyError
        to avoid silently resetting its EMA.
        """
        if slug in self.progress:
            raise KeyError(
                f"track {slug!r} already present; use reset() if you want to "
                "clear its progress"
            )
        self.track_slugs = [*self.track_slugs, slug]
        self.progress[slug] = self.config.cold_start_progress

    @property
    def n_tracks(self) -> int:
        return len(self.progress)

    # ------------------------------------------------------------------
    # Checkpoint serialization.
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialize progress EMA + RNG state for checkpoint resume.

        The RNG state is preserved so post-resume ``sample()`` sequences
        match the uninterrupted run (under a fixed seed).
        """
        return {
            "progress": dict(self.progress),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from ``state_dict()``. Silently keeps tracks that weren't
        in the saved state at cold-start progress (e.g., user added a savestate
        between the ckpt and the resume)."""
        for slug, p in state["progress"].items():
            if slug in self.progress:
                self.progress[slug] = float(p)
        self._rng.bit_generator.state = state["rng_state"]


def construct_from_available(
    savestate_slugs: Iterable[str],
    config: TrackSamplerConfig | None = None,
    seed: int | None = None,
) -> ProgressWeightedTrackSampler:
    """Convenience: build a sampler keyed by the slugs currently on disk.

    The training loop typically calls ``env.available_tracks()`` (which
    returns the list of ``{slug}.sav`` files in ``data/savestates/``) and
    feeds the result into this function to get a sampler that covers
    exactly the tracks we can actually run.
    """
    return ProgressWeightedTrackSampler(
        track_slugs=list(savestate_slugs),
        config=config or TrackSamplerConfig(),
        seed=seed,
    )
