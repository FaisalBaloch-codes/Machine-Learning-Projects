"""Multi-factor event-impact labelling: spike / crash / neutral.

A label describes the MARKET REACTION that followed a news event:
"the stock exhibited a statistically unusual market reaction following this
news event". It is an association, not evidence that the article caused the move.

Signals (all configurable in .env)
----------------------------------
For each horizon h in LABEL_HORIZONS (default 5, 15, 30, 60 minutes):
  return_h          (close[T0+h] - close[T0]) / close[T0]
  abnormal z_h      return_h / (sigma_1m * sqrt(h))   sigma_1m = pre-news per-minute
                    volatility of the same stock (prior sessions + same-day pre-news bars)
  volume_ratio      event-window volume / same clock-time volume in prior sessions

A horizon QUALIFIES in direction d (+ spike / - crash) when ALL hold:
  1. |z_h| >= ABNORMAL_RETURN_Z                  (statistically unusual for THIS stock)
  2. |return_h| >= MIN_ABS_RETURN                (economically non-trivial)
  3. confirmation: volume_ratio >= VOLUME_SPIKE_RATIO, or (unless
     REQUIRE_VOLUME_CONFIRMATION) return beyond SPIKE/CRASH_RETURN_THRESHOLD
  4. direction consistency: share of available horizons whose return has the
     same sign >= DIRECTION_CONSISTENCY_MIN (for overnight news the sign of the
     cumulative move from the last pre-news close, i.e. gap compounded with
     each post-open return)
Overnight news (next_market_open) additionally tests the opening gap
(gap_return, z vs sigma_1m*sqrt(390)) when LABEL_INCLUDE_OVERNIGHT_GAP=true.

Decision: if qualifying horizons exist in only one direction -> spike/crash
using the strongest (max |z|) horizon; if both directions qualify -> neutral
(conflicting reaction); otherwise neutral. Events without usable market data
or with too little baseline history get NO label (empty), never a guess.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import Settings

LABEL_COLUMNS = ["label", "label_confidence", "label_reason", "label_horizon_minutes", "abnormal_return",
                 "direction_consistency"]


def _num(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def label_event(row: dict, s: Settings) -> dict:
    empty = {"label": "", "label_confidence": math.nan, "label_horizon_minutes": math.nan,
             "abnormal_return": math.nan, "direction_consistency": math.nan}
    if row.get("market_alignment") in (None, "", "unavailable") or (isinstance(row.get("market_alignment"), float)):
        return {**empty, "label_reason": f"Not labelled: market data unavailable ({row.get('market_unavailable_reason') or 'no aligned candle'})"}
    sigma = _num(row.get("volatility"))
    if not sigma > 0 or _num(row.get("baseline_return_count")) < s.min_baseline_bars:
        return {**empty, "label_reason": f"Not labelled: insufficient pre-news baseline "
                                          f"({int(_num(row.get('baseline_return_count')) or 0)} returns < {s.min_baseline_bars})"}

    horizons = []
    for h in s.label_horizons:
        r, z = _num(row.get(f"return_{h}m")), _num(row.get(f"abnormal_return_{h}m"))
        if not math.isnan(r) and not math.isnan(z):
            horizons.append((f"{h}-minute", h, r, z))
    gap_r, gap_z = _num(row.get("gap_return")), _num(row.get("gap_abnormal"))
    if s.label_include_overnight_gap and row.get("market_alignment") == "next_market_open" \
            and not math.isnan(gap_r) and not math.isnan(gap_z):
        horizons.append(("opening-gap", 0, gap_r, gap_z))
    if not horizons:
        return {**empty, "label_reason": "Not labelled: no post-news window available before the session close"}

    vr = _num(row.get("volume_ratio"))
    vol_ok = not math.isnan(vr) and vr >= s.volume_spike_ratio
    if horizons[-1][0] == "opening-gap":
        # Overnight news: direction is judged on the price relative to the last pre-news
        # close, i.e. the gap compounded with each post-open return.
        g = horizons[-1][2]
        signs = [np.sign((1 + g) * (1 + r) - 1) for name, _, r, _ in horizons if name != "opening-gap"]
        signs = [x for x in signs + [np.sign(g)] if x != 0]
    else:
        signs = [np.sign(r) for _, _, r, _ in horizons if r != 0]

    def consistency(direction: float) -> float:
        return sum(1 for x in signs if x == direction) / len(signs) if signs else 0.0

    qualifying, near_misses = [], []
    for name, h, r, z in horizons:
        direction = 1.0 if r > 0 else -1.0
        abnormal = abs(z) >= s.abnormal_return_z
        meaningful = abs(r) >= s.min_abs_return
        large = r >= s.spike_return_threshold if direction > 0 else r <= s.crash_return_threshold
        confirmed = vol_ok or (large and not s.require_volume_confirmation)
        cons = consistency(direction)
        if abnormal and meaningful and confirmed and cons >= s.direction_consistency_min:
            qualifying.append((name, h, r, z, direction, large, cons))
        elif abnormal:
            missing = []
            if not meaningful:
                missing.append(f"|return| < {s.min_abs_return:.2%}")
            if not confirmed:
                missing.append(f"volume {_fmt_vr(vr)} < {s.volume_spike_ratio:g}x and return inside "
                               f"{s.crash_return_threshold:+.1%}/{s.spike_return_threshold:+.1%}")
            if cons < s.direction_consistency_min:
                missing.append(f"direction consistent in only {cons:.0%} of horizons")
            near_misses.append((name, r, z, "; ".join(missing)))

    directions = {q[4] for q in qualifying}
    best_all = max(horizons, key=lambda x: abs(x[3]))

    if len(directions) > 1:
        up = max((q for q in qualifying if q[4] > 0), key=lambda q: abs(q[3]))
        dn = max((q for q in qualifying if q[4] < 0), key=lambda q: abs(q[3]))
        return {"label": "neutral", "label_confidence": 0.4, "label_horizon_minutes": best_all[1],
                "abnormal_return": best_all[3], "direction_consistency": math.nan,
                "label_reason": f"Conflicting reaction: abnormal rise at {up[0]} ({up[2]:+.2%}, z={up[3]:.1f}) "
                                f"and abnormal fall at {dn[0]} ({dn[2]:+.2%}, z={dn[3]:.1f}); treated as neutral"}

    if qualifying:
        name, h, r, z, direction, large, cons = max(qualifying, key=lambda q: abs(q[3]))
        z_strength = _clip01((abs(z) - s.abnormal_return_z) / s.abnormal_return_z)
        vol_strength = _clip01(vr / (2 * s.volume_spike_ratio)) if not math.isnan(vr) else 0.0
        thr = s.spike_return_threshold if direction > 0 else abs(s.crash_return_threshold)
        ret_strength = _clip01(abs(r) / thr)
        conf = 0.5 + 0.5 * (0.4 * z_strength + 0.25 * vol_strength + 0.2 * ret_strength + 0.15 * cons)
        label = "spike" if direction > 0 else "crash"
        strength = "Strong" if conf >= 0.8 else "Moderate"
        vol_txt = f"{vr:.1f}x normal volume" if not math.isnan(vr) else "volume baseline unavailable"
        reason = (f"{strength} {'positive' if direction > 0 else 'negative'} {name} abnormal return "
                  f"({r:+.2%}, z={z:+.1f}) with {vol_txt}; direction consistent in {cons:.0%} of horizons"
                  f"{'; exceeds absolute return threshold' if large else ''}")
        return {"label": label, "label_confidence": round(conf, 4), "label_reason": reason,
                "label_horizon_minutes": h, "abnormal_return": z, "direction_consistency": round(cons, 4)}

    ratio = abs(best_all[3]) / s.abnormal_return_z
    conf = 0.5 + 0.5 * (1 - _clip01(ratio))
    if near_misses:
        nm = max(near_misses, key=lambda x: abs(x[2]))
        reason = (f"Neutral: {nm[0]} move {nm[1]:+.2%} (z={nm[2]:+.1f}) was statistically unusual but not "
                  f"confirmed ({nm[3]})")
    else:
        reason = (f"Neutral: no statistically unusual move; largest |z|={abs(best_all[3]):.1f} "
                  f"(< {s.abnormal_return_z:g}) at {best_all[0]} ({best_all[2]:+.2%}); volume {_fmt_vr(vr)}")
    return {"label": "neutral", "label_confidence": round(conf, 4), "label_reason": reason,
            "label_horizon_minutes": best_all[1], "abnormal_return": best_all[3], "direction_consistency": math.nan}


def _fmt_vr(vr: float) -> str:
    return "n/a" if math.isnan(vr) else f"{vr:.1f}x"


def label_frame(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    labels = [label_event(r, s) for r in df.to_dict("records")]
    lab = pd.DataFrame(labels, index=df.index).reindex(columns=LABEL_COLUMNS)
    return pd.concat([df, lab], axis=1)


def run_label_stage(settings: Settings) -> dict:
    import json
    import logging

    from config.settings import LABEL_SETTING_NAMES
    from storage.checkpoint import Checkpoint
    from storage.csv_writer import AtomicCSVWriter, iter_csv_chunks

    log = logging.getLogger(__name__)
    cp = Checkpoint(settings.checkpoint_dir / "labeling.json")
    src = settings.market_features_path
    if not src.exists():
        log.warning("No market features found - run --market first.")
        return {"status": "no_input"}
    sig = {"config": settings.section_hash(*LABEL_SETTING_NAMES), "input_size": src.stat().st_size,
           "input_mtime": int(src.stat().st_mtime)}
    if cp.get("signature") == sig and settings.labeled_events_path.exists():
        log.info("Labels are up to date (same inputs and label settings) - skipping.")
        return {"status": "cached", **cp.get("counts", {})}

    writer = AtomicCSVWriter(settings.labeled_events_path)
    counts: dict[str, int] = {}
    try:
        for chunk in iter_csv_chunks(src, settings.chunk_size, dtype=None):
            out = label_frame(chunk, settings)
            for k, v in out["label"].fillna("").replace("", "unlabelled").value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
            writer.write(out)
        writer.commit()
    except BaseException:
        writer.abort()
        raise
    cp.update({"signature": sig, "counts": counts})
    log.info("Labels: %s", json.dumps(counts))
    return {"status": "completed", **counts}
