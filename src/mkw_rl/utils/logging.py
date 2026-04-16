"""Training run logging: wandb if available, CSV fallback.

The choice of backend is automatic — if ``WANDB_API_KEY`` is in the
environment, we use wandb. Otherwise we write tab-separated metrics
to a local CSV. Downstream training code just calls ``logger.log({...})``
and doesn't care which.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class CsvLogger:
    """Minimal CSV logger with fixed columns.

    Behavior:
    * First ``log()`` call determines the column set. These columns are
      written as a header and the file is kept open.
    * Subsequent calls MUST provide the same keys. Missing keys get empty
      strings. Extra keys trigger a ONE-TIME warning per new key and are
      dropped — we do NOT silently extend columns mid-file because that
      would desync header and data rows (H-3 audit fix).
    * File is opened in append mode if the path already exists; otherwise
      written fresh. A header row is written only when the file is new
      (L-7 audit fix — prior-run data is not clobbered).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._columns: list[str] = []
        self._file = None
        self._writer = None
        self._warned_extra_keys: set[str] = set()

    def log(self, metrics: dict[str, Any]) -> None:
        if self._file is None:
            self._columns = list(metrics.keys())
            is_new = not self.path.exists() or self.path.stat().st_size == 0
            self._file = self.path.open("a", newline="")
            self._writer = csv.writer(self._file, delimiter="\t")
            if is_new:
                self._writer.writerow(self._columns)
        # Warn about (and drop) unknown keys — do not extend columns mid-file.
        for k in metrics:
            if k not in self._columns and k not in self._warned_extra_keys:
                log.warning(
                    "CsvLogger: ignoring unknown key %r (columns were fixed on first log() call)",
                    k,
                )
                self._warned_extra_keys.add(k)
        assert self._writer is not None
        row = [metrics.get(c, "") for c in self._columns]
        self._writer.writerow(row)
        assert self._file is not None
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class WandbLogger:
    """wandb-backed logger.

    Note: wandb.init is called in __init__, so creating the logger already
    starts a run. The ``wandb`` import is deferred until construction so
    the CSV path never incurs the wandb import cost.
    """

    def __init__(self, project: str, config: dict[str, Any] | None = None) -> None:
        import wandb  # deferred — CSV path never imports wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, config=config or {}, reinit=True)

    def log(self, metrics: dict[str, Any]) -> None:
        self._wandb.log(metrics)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def make_logger(
    project: str,
    csv_fallback_path: Path | str,
    config: dict[str, Any] | None = None,
) -> CsvLogger | WandbLogger:
    """Return the best available logger based on env and installed packages."""
    if os.environ.get("WANDB_API_KEY"):
        try:
            return WandbLogger(project=project, config=config)
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb init failed (%s); falling back to CSV", exc)
    return CsvLogger(csv_fallback_path)
