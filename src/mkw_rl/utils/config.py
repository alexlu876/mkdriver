"""Thin YAML config loader.

Deliberately minimal. No Hydra, no DictConfig, no overrides syntax.
Just load YAML → dict and hand it to the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config file into a nested dict."""
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def get_nested(cfg: dict[str, Any], keys: str, default: Any = None) -> Any:
    """Look up a dotted key path in a nested dict.

    Example: ``get_nested(cfg, "data.batch_size", 16)``.
    """
    parts = keys.split(".")
    cur: Any = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur
