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
    """Minimal CSV logger.

    Appends one row per ``log()`` call. First call determines the column
    order; subsequent calls must use the same keys (missing columns are
    filled with empty strings, extra columns are appended).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._columns: list[str] = []
        self._file = None
        self._writer = None

    def log(self, metrics: dict[str, Any]) -> None:
        if self._file is None:
            self._columns = list(metrics.keys())
            self._file = self.path.open("w", newline="")
            self._writer = csv.writer(self._file, delimiter="\t")
            self._writer.writerow(self._columns)
        # Extend columns if new keys appeared.
        for k in metrics:
            if k not in self._columns:
                self._columns.append(k)
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
    """wandb-backed logger. Initialized lazily on first ``log()``."""

    def __init__(self, project: str, config: dict[str, Any] | None = None) -> None:
        import wandb  # imported lazily so CSV path doesn't incur the import

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
