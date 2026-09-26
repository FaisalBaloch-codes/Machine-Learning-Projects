"""Vectorised timestamp helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_seconds(values) -> np.ndarray:
    """UTC epoch seconds (int64) for a sequence of timestamps/ISO strings."""
    ts = pd.to_datetime(pd.Series(values), utc=True)
    return ((ts - _EPOCH) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
