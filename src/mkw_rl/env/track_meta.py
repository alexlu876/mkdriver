"""Per-track metadata loader.

Reads ``data/track_metadata.yaml`` (see that file's docstring for schema).
The loaded metadata drives the v2 reward function's variable-checkpoint-per-track
behavior: ``n_checkpoints_per_lap = round(100 × wr_seconds / 60)``.

Pure Python + PyYAML only — intentionally no heavy deps so this module is
importable from inside Dolphin's embedded interpreter when the slave script
forwards our src/ directory onto ``sys.path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# NOTE: yaml is imported lazily inside _load_yaml so that the slave-side
# ``dolphin_script.py`` can import ``TrackMetadata`` (the dataclass) from this
# module without needing PyYAML in Dolphin's embedded Python. The slave only
# constructs ``TrackMetadata`` from dicts forwarded via IPC from the master.
_DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[3] / "data" / "track_metadata.yaml"


@dataclass(frozen=True)
class TrackMetadata:
    """Immutable per-track record."""

    slug: str
    name: str
    cup: str
    wr_seconds: float
    wr_category: str  # "non_glitch" or "shortcut"
    laps: int

    @property
    def n_checkpoints_per_lap(self) -> int:
        """v2 formula: 100 × WR minutes, rounded to nearest int."""
        return round(100 * self.wr_seconds / 60)

    @property
    def n_checkpoints_total(self) -> int:
        return self.n_checkpoints_per_lap * self.laps


def _load_yaml(path: Path) -> dict:
    import yaml  # lazy — see module docstring

    if not path.exists():
        raise FileNotFoundError(f"track metadata file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_track_metadata(path: Path | str | None = None) -> dict[str, TrackMetadata]:
    """Load the YAML into a ``{slug: TrackMetadata}`` dict.

    If ``path`` is None, defaults to ``<project_root>/data/track_metadata.yaml``.
    """
    resolved = Path(path) if path is not None else _DEFAULT_METADATA_PATH
    raw = _load_yaml(resolved)

    out: dict[str, TrackMetadata] = {}
    for slug, fields in raw.items():
        # Schema is enforced minimally — require the fields we use; tolerate
        # additions (e.g., `notes`) for forward compat.
        for required in ("name", "cup", "wr_seconds", "wr_category", "laps"):
            if required not in fields:
                raise ValueError(f"track {slug!r} missing required field {required!r}")
        if fields["wr_category"] not in ("non_glitch", "shortcut"):
            raise ValueError(
                f"track {slug!r} has invalid wr_category {fields['wr_category']!r} "
                "(must be 'non_glitch' or 'shortcut')"
            )
        if fields["laps"] < 1:
            raise ValueError(f"track {slug!r} has non-positive laps: {fields['laps']}")
        if fields["wr_seconds"] <= 0:
            raise ValueError(f"track {slug!r} has non-positive wr_seconds: {fields['wr_seconds']}")

        out[slug] = TrackMetadata(
            slug=slug,
            name=fields["name"],
            cup=fields["cup"],
            wr_seconds=float(fields["wr_seconds"]),
            wr_category=fields["wr_category"],
            laps=int(fields["laps"]),
        )
    return out


@lru_cache(maxsize=1)
def _cached_default_metadata() -> dict[str, TrackMetadata]:
    return load_track_metadata()


def checkpoint_count_for_track(slug: str, path: Path | str | None = None) -> int:
    """Lookup helper — returns ``n_checkpoints_per_lap`` for a track slug.

    Uses a module-level cache for the default metadata file to avoid
    re-parsing the YAML on every call (the env's reward tracker calls this
    once per reset).
    """
    meta = load_track_metadata(path) if path is not None else _cached_default_metadata()
    if slug not in meta:
        raise KeyError(f"unknown track slug: {slug!r} (valid: {sorted(meta)})")
    return meta[slug].n_checkpoints_per_lap
