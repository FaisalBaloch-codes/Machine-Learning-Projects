"""Incremental, crash-safe CSV writing helpers.

Design
------
* Rows are appended in batches and every batch is flushed + fsync'ed.
* After each batch the caller stores the new file size in its checkpoint
  ("committed size"). On restart, any bytes beyond the committed size (e.g. a
  half-written row from a hard kill) are moved to an `.orphan` file and the CSV
  is truncated back to the last committed row. Nothing is silently deleted.
* Derived files (dedup output, final dataset, ...) are written to a temporary
  file and atomically moved into place.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

# Some fields (full article content) can be large.
csv.field_size_limit(min(2**31 - 1, 50 * 1024 * 1024))


def read_header(path: Path) -> list[str] | None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), None)


class IncrementalCSVWriter:
    def __init__(self, path: Path, columns: list[str], committed_size: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._repair(committed_size)
        existing = read_header(self.path)
        if existing:
            missing = [c for c in columns if c not in existing]
            if missing:
                log.warning("%s has an older header; columns %s will not be written", self.path.name, missing)
            self.columns = existing
        else:
            self.columns = list(columns)

    def _repair(self, committed_size: int | None) -> None:
        if committed_size is None or not self.path.exists():
            return
        size = self.path.stat().st_size
        if size > committed_size:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            orphan = self.path.with_name(f"{self.path.name}.{stamp}.orphan")
            with open(self.path, "rb+") as fh:
                fh.seek(committed_size)
                tail = fh.read()
                with open(orphan, "wb") as out:
                    out.write(tail)
                fh.truncate(committed_size)
            log.warning("%s: %d uncommitted bytes after last checkpoint moved to %s (they will be re-collected)",
                        self.path.name, len(tail), orphan.name)
        elif size < committed_size:
            log.warning("%s is smaller than the checkpoint expects (%d < %d); continuing with the file as-is",
                        self.path.name, size, committed_size)

    def write_rows(self, rows: Iterable[dict]) -> int:
        """Append rows; returns the file size after the write (the new committed size)."""
        rows = list(rows)
        with self.lock:
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore",
                                        quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                if new_file:
                    writer.writeheader()
                for row in rows:
                    writer.writerow({k: _cell(v) for k, v in row.items()})
                fh.flush()
                os.fsync(fh.fileno())
            return self.path.stat().st_size

    def size(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return value


def read_column_set(path: Path, column: str, chunksize: int = 200_000) -> set[str]:
    """Collect the distinct values of one column without loading the whole file."""
    path = Path(path)
    header = read_header(path)
    if not header or column not in header:
        return set()
    values: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[column], dtype=str, chunksize=chunksize,
                             keep_default_na=False, on_bad_lines="warn"):
        values.update(v for v in chunk[column].tolist() if v)
    return values


def iter_csv_chunks(path: Path, chunksize: int, usecols: list[str] | None = None, dtype=str):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    yield from pd.read_csv(path, chunksize=chunksize, usecols=usecols, dtype=dtype,
                           keep_default_na=False, na_values=[""], on_bad_lines="warn")


class AtomicCSVWriter:
    """Stream chunks into a temp file, then atomically replace the target."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        self._header_written = False
        self._columns: list[str] | None = None
        self.rows = 0
        if self.tmp.exists():
            self.tmp.unlink()

    def write(self, df: pd.DataFrame) -> None:
        if self._columns is None:
            self._columns = list(df.columns)
        df = df.reindex(columns=self._columns)
        df.to_csv(self.tmp, mode="a", header=not self._header_written, index=False,
                  encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        self._header_written = True
        self.rows += len(df)

    def commit(self, empty_columns: list[str] | None = None) -> None:
        if not self._header_written:
            pd.DataFrame(columns=empty_columns or self._columns or []).to_csv(self.tmp, index=False)
        with open(self.tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        if self.tmp.exists():
            self.tmp.unlink()


def atomic_write_df(df: pd.DataFrame, path: Path) -> None:
    writer = AtomicCSVWriter(path)
    writer.write(df)
    writer.commit()


def archive_file(path: Path) -> Path | None:
    """Rename an output file out of the way instead of deleting it (used by --force)."""
    path = Path(path)
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.stem}.{stamp}.archived{path.suffix}")
    os.replace(path, target)
    log.info("Archived %s -> %s", path.name, target.name)
    return target
