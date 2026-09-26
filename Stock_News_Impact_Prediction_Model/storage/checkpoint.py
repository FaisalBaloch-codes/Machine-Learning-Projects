"""Atomic JSON checkpoints.

A checkpoint is written to a temporary file, fsync'ed and then moved over the
old file with os.replace (atomic on Windows, macOS and Linux). The previous
version is kept as `<name>.json.bak` so a corrupted file can be recovered.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding=encoding, newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    # On Windows, antivirus / search indexing can hold the target open for a moment
    # right after it is written, making os.replace fail with "Access is denied".
    delay = 0.05
    for attempt in range(12):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 11:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


class Checkpoint:
    """Thread-safe, atomically persisted dictionary."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        for candidate in (self.path, self.path.with_name(self.path.name + ".bak")):
            if candidate.exists():
                try:
                    with open(candidate, encoding="utf-8") as fh:
                        data = json.load(fh)
                    if candidate != self.path:
                        log.warning("Recovered checkpoint from backup %s", candidate)
                    return data
                except (json.JSONDecodeError, OSError) as exc:
                    log.error("Could not read checkpoint %s: %s", candidate, exc)
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self.data.get(key, default)

    def set(self, key: str, value: Any, save: bool = True) -> None:
        with self.lock:
            self.data[key] = value
            if save:
                self.save()

    def update(self, values: dict[str, Any], save: bool = True) -> None:
        with self.lock:
            self.data.update(values)
            if save:
                self.save()

    def save(self) -> None:
        with self.lock:
            self.data["_updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if self.path.exists():
                try:
                    shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
                except OSError:
                    pass
            atomic_write_text(self.path, json.dumps(self.data, indent=2, sort_keys=True, default=str))

    def reset(self) -> None:
        """Archive the current checkpoint (never silently deleted) and start empty."""
        with self.lock:
            if self.path.exists():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.replace(self.path, self.path.with_name(f"{self.path.stem}.{stamp}.archived.json"))
            self.data = {}
            self.save()
