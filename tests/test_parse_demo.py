"""Tests for the sidecar-reading helper in scripts/parse_demo.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_parse_demo():
    """Import scripts/parse_demo.py as a module."""
    path = Path(__file__).parent.parent / "scripts" / "parse_demo.py"
    spec = importlib.util.spec_from_file_location("parse_demo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestSavestateSidecar:
    def test_reads_skip_first_n(self, tmp_path: Path) -> None:
        pd = _load_parse_demo()
        sidecar = tmp_path / "luigi.json"
        sidecar.write_text(
            json.dumps(
                {
                    "game_id": "RMCP01",
                    "track": "luigi_circuit",
                    "vi_count": 3427,
                    "skip_first_n": 42,
                }
            )
        )
        assert pd._skip_first_n_from_savestate(sidecar) == 42

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        pd = _load_parse_demo()
        sidecar = tmp_path / "bad.json"
        sidecar.write_text(json.dumps({"game_id": "RMCP01"}))  # no skip_first_n
        with pytest.raises(KeyError, match="skip_first_n"):
            pd._skip_first_n_from_savestate(sidecar)

    def test_zero_is_valid(self, tmp_path: Path) -> None:
        pd = _load_parse_demo()
        sidecar = tmp_path / "ok.json"
        sidecar.write_text(json.dumps({"skip_first_n": 0}))
        assert pd._skip_first_n_from_savestate(sidecar) == 0
