# -*- coding: utf-8 -*-
"""
FinBERT news -> market-effect classifier: complete training & evaluation pipeline.

    ml_ready_dataset.csv
        -> inspection -> cleaning / title-level de-duplication -> time-based split
        -> TF-IDF + Logistic Regression baseline -> FinBERT fine-tuning
        -> evaluation, plots, error & confidence analysis -> saved model + report

Designed for Google Colab (GPU), also runs on CPU (slowly) for a smoke test.

HOW TO RUN IN COLAB (see README.md for screenshots-free step-by-step):
    1. Runtime -> Change runtime type -> GPU (T4 is enough).
    2. In a notebook cell:
           from google.colab import drive
           drive.mount('/content/drive')
           !pip install -q -r /content/drive/MyDrive/finbert_market_effect/requirements.txt
    3. Edit DATA_PATH / OUTPUT_DIR below (or pass --data-path / --output-dir), then:
           !python /content/drive/MyDrive/finbert_market_effect/train_finbert.py
       The file is also split into "# %%" cells, so it can be pasted into a notebook cell-by-cell.

THE MODEL INPUT IS THE NEWS TITLE ONLY. Future prices, returns, volume and every
label-generation field are used for NOTHING except the target label (see section 3).
"""

# %% ==========================================================================
# 1. CONFIGURATION - edit these values (or override the main ones on the command line)
# =============================================================================
DATA_PATH = "/content/drive/MyDrive/finbert_market_effect/ml_ready_dataset.csv"
OUTPUT_DIR = "/content/drive/MyDrive/finbert_market_effect"   # all outputs go here (Drive = survives disconnects)

# --- dataset columns (the script checks them and tells you if they do not exist) ---
TITLE_COLUMN = "title"
LABEL_COLUMN = "label"
TIMESTAMP_COLUMN = "news_published_timestamp_utc"   # used for the time-based split; None -> stratified split
TICKER_COLUMN = "ticker"                             # optional, only used for reports / ticker_title mode

# --- label mapping: position in the list = class id used by the model -------
LABELS = ["neutral", "spike", "crash"]              # 0 = neutral, 1 = spike, 2 = crash

# --- model input -------------------------------------------------------------
# "title"        : model sees only the headline (what you asked for). Rows that repeat the
#                  same headline for several tickers are merged into ONE example.
# "ticker_title" : model sees "TICKER: headline"; one example per (headline, ticker).
INPUT_MODE = "title"
# When one headline has several rows with different labels (different tickers / dates):
#   majority vote; exact ties are dropped as ambiguous (reported).
MIN_TITLE_CHARS = 8                                  # shorter titles are treated as malformed

# --- split (chronological) ---------------------------------------------------
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# --- model / training hyper-parameters ---------------------------------------
MODEL_NAME = "ProsusAI/finbert"
REUSE_SENTIMENT_HEAD = True     # start spike/crash/neutral from FinBERT's positive/negative/neutral head
MAX_LENGTH = 64                 # headlines are short; truncation rate is printed so you can check
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1 # raise (and lower batch size) only if you run out of GPU memory
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EVALS_PER_EPOCH = 2             # validation checks per epoch (used for early stopping / best checkpoint)
EARLY_STOPPING_PATIENCE = 3     # number of validation checks without macro-F1 improvement
CLASS_WEIGHTING = "balanced"    # "balanced" | "sqrt" | "none"   (see section 6 for the reasoning)
SEED = 42
RESUME_FROM_CHECKPOINT = True   # continue automatically after a Colab disconnect

# --- quick functional test (CPU friendly): tiny sample, few steps ------------
SMOKE_TEST = False
SMOKE_TEST_ROWS = 3000
SMOKE_TEST_MAX_STEPS = 20

# %% ==========================================================================
# 2. SETUP: imports, command-line overrides, device, seeds
# =============================================================================
import argparse
import html
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument("--data-path")
_parser.add_argument("--output-dir")
_parser.add_argument("--epochs", type=float)
_parser.add_argument("--batch-size", type=int)
_parser.add_argument("--smoke-test", action="store_true")
_parser.add_argument("--model-name")
_args, _ = _parser.parse_known_args()        # parse_known_args: also works inside Jupyter/Colab cells
DATA_PATH = _args.data_path or DATA_PATH
OUTPUT_DIR = _args.output_dir or OUTPUT_DIR
NUM_EPOCHS = _args.epochs or NUM_EPOCHS
TRAIN_BATCH_SIZE = _args.batch_size or TRAIN_BATCH_SIZE
SMOKE_TEST = SMOKE_TEST or _args.smoke_test
MODEL_NAME = _args.model_name or MODEL_NAME

import torch

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

OUT = Path(OUTPUT_DIR)
RESULTS = OUT / "results"
MODEL_DIR = OUT / "model" / "finbert_market_effect"
CKPT_DIR = OUT / "checkpoints"
for d in (OUT, RESULTS, MODEL_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
DISPLAY = [l.capitalize() for l in LABELS]
RUN_INFO: dict = {"started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "device": str(DEVICE),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "smoke_test": SMOKE_TEST}
if SMOKE_TEST:
    print(f"\n*** SMOKE TEST MODE: {SMOKE_TEST_ROWS} rows, {SMOKE_TEST_MAX_STEPS} training steps. "
          "Results are NOT meaningful. ***\n")

# %% ==========================================================================
# 3. LOAD & INSPECT THE DATASET
# =============================================================================
# ANTI-LEAKAGE: we read ONLY the title, the label, the timestamp (for splitting) and the
# ticker (for reports). Columns such as volatility, pre_news_*, sentiment, label_confidence
# are never loaded, and future returns / OHLCV are not in this file at all. The label itself
# was created from post-news price movement - that is the TARGET, never an input.
if not Path(DATA_PATH).exists():
    sys.exit(f"DATA_PATH not found: {DATA_PATH}\nUpload ml_ready_dataset.csv and set DATA_PATH (or --data-path).")

header = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
print(f"Columns in file ({len(header)}): {header}")


def resolve_column(configured, candidates, required, purpose):
    """Use the configured column if it exists, otherwise try common alternatives and say so."""
    if configured and configured in header:
        return configured
    for c in candidates:
        if c in header:
            print(f"NOTE: column '{configured}' not found; using '{c}' for {purpose}. "
                  f"Set the corresponding *_COLUMN in the config to silence this.")
            return c
    if required:
        sys.exit(f"ERROR: no column for {purpose}. Configured '{configured}' is not in the file. "
                 f"Available columns: {header}. Edit the config section.")
    print(f"NOTE: no {purpose} column found - continuing without it.")
    return None


TITLE_COL = resolve_column(TITLE_COLUMN, ["title", "headline", "news_title", "text"], True, "news titles")
LABEL_COL = resolve_column(LABEL_COLUMN, ["label", "target", "class", "market_effect"], True, "labels")
TS_COL = resolve_column(TIMESTAMP_COLUMN, ["news_published_timestamp_utc", "published_at", "timestamp",
                                           "created_at", "news_date", "date"], False, "timestamps")
TICKER_COL = resolve_column(TICKER_COLUMN, ["ticker", "symbol"], False, "tickers")
if INPUT_MODE == "ticker_title" and not TICKER_COL:
    sys.exit("INPUT_MODE='ticker_title' needs a ticker column.")

usecols = [c for c in (TITLE_COL, LABEL_COL, TS_COL, TICKER_COL) if c]
raw = pd.read_csv(DATA_PATH, usecols=usecols, dtype=str, keep_default_na=False, na_values=[""])
raw = raw.rename(columns={TITLE_COL: "title", LABEL_COL: "label",
                          **({TS_COL: "timestamp"} if TS_COL else {}),
                          **({TICKER_COL: "ticker"} if TICKER_COL else {})})

inspection: dict = {"rows": len(raw), "columns_in_file": header, "columns_used": usecols}
inspection["missing_values"] = {k: int(v) for k, v in raw.isna().sum().items()}
inspection["duplicate_rows"] = int(raw.duplicated().sum())
inspection["duplicate_titles"] = int(raw["title"].duplicated().sum())
if "ticker" in raw:
    inspection["duplicate_title_ticker"] = int(raw.duplicated(["title", "ticker"]).sum())
    inspection["unique_tickers"] = int(raw["ticker"].nunique())
    inspection["top_20_tickers"] = raw["ticker"].value_counts().head(20).to_dict()
norm_label = raw["label"].astype(str).str.strip().str.lower()
inspection["label_distribution_raw"] = norm_label.value_counts(dropna=False).to_dict()
inspection["invalid_labels"] = int((~norm_label.isin(LABELS)).sum())
t = raw["title"].fillna("").astype(str)
inspection["empty_or_missing_titles"] = int((t.str.strip() == "").sum())
inspection["title_length_chars"] = t.str.len().describe().round(1).to_dict()
inspection["title_length_words"] = t.str.split().str.len().describe().round(1).to_dict()
vc = norm_label[norm_label.isin(LABELS)].value_counts()
inspection["imbalance_ratio_majority_to_minority"] = round(float(vc.max() / vc.min()), 2) if len(vc) else None
if TS_COL:
    ts_tmp = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    inspection["invalid_timestamps"] = int(ts_tmp.isna().sum())
    inspection["timestamp_range"] = [str(ts_tmp.min()), str(ts_tmp.max())]

print("\n=== DATASET INSPECTION ===")
for k, v in inspection.items():
    if k not in ("columns_in_file",):
        print(f"{k}: {v}")
if inspection["invalid_labels"]:
    bad = norm_label[~norm_label.isin(LABELS)].value_counts().head(10).to_dict()
    print(f"WARNING: {inspection['invalid_labels']} rows have labels outside {LABELS}: {bad}. "
          "They will be dropped. If these are valid classes, edit LABELS.")

# %% ==========================================================================
# 4. CLEANING (light, Transformer-friendly) + DE-DUPLICATION
# =============================================================================
# We do NOT remove stop-words, stem, lowercase (the tokenizer handles case), or strip numbers,
# tickers, '$', '%' - they carry financial meaning. We only repair text and drop unusable rows.
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f​-‏  ﻿]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                         "“": '"', "”": '"', "„": '"', "″": '"',
                         "–": "-", "—": "-", "−": "-", " ": " "})


def clean_title(text) -> str:
    """Identical function is used in predict.py - keep them in sync."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = html.unescape(str(text))
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    s = _CTRL.sub(" ", s)
    s = re.sub(r"<[^>]+>", " ", s)          # stray HTML tags
    return re.sub(r"\s+", " ", s).strip()


def title_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


df = raw.copy()
report_clean: dict = {"input_rows": len(df)}
df["label"] = df["label"].astype(str).str.strip().str.lower()
mask_label = df["label"].isin(LABELS)
report_clean["dropped_invalid_label"] = int((~mask_label).sum())
df = df[mask_label]
df["title"] = df["title"].map(clean_title)
empty = df["title"] == ""
report_clean["dropped_missing_or_empty_title"] = int(empty.sum())
df = df[~empty]
malformed = (df["title"].str.len() < MIN_TITLE_CHARS) | ~df["title"].str.contains(r"[A-Za-z]", regex=True)
report_clean["dropped_malformed_title"] = int(malformed.sum())   # < MIN_TITLE_CHARS or no Latin letters
report_clean["malformed_examples"] = df.loc[malformed, "title"].head(10).tolist()
df = df[~malformed]
if "timestamp" in df:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = df["timestamp"].isna()
    report_clean["dropped_invalid_timestamp"] = int(bad_ts.sum())
    df = df[~bad_ts]
before = len(df)
df = df.drop_duplicates()
report_clean["dropped_exact_duplicate_rows"] = before - len(df)

# One training example per unique input text. Identical inputs with different targets would
# only add noise, and a headline must never appear in both train and test (leakage).
df["key"] = df["title"].map(title_key)
if INPUT_MODE == "ticker_title":
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper()
    df["key"] = df["ticker"] + "|" + df["key"]
agg = {"title": "first", "label": list}
if "timestamp" in df:
    agg["timestamp"] = "min"
if "ticker" in df:
    agg["ticker"] = lambda x: ";".join(sorted({str(v) for v in x if isinstance(v, str) and v}))
df = df.sort_values("timestamp" if "timestamp" in df else "key", kind="stable")
grp = df.groupby("key", sort=False).agg(agg).reset_index()


def resolve(labels: list) -> tuple[str | None, float]:
    counts = pd.Series(labels).value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return None, counts.iloc[0] / len(labels)      # exact tie -> ambiguous
    return counts.index[0], counts.iloc[0] / len(labels)


res = grp["label"].map(resolve)
grp["n_source_rows"] = grp["label"].map(len)
grp["had_conflict"] = grp["label"].map(lambda l: len(set(l)) > 1)
grp["label_agreement"] = [r[1] for r in res]
grp["label"] = [r[0] for r in res]
report_clean["rows_before_dedup"] = len(df)
report_clean["unique_inputs"] = len(grp)
report_clean["inputs_with_conflicting_labels"] = int(grp["had_conflict"].sum())
report_clean["dropped_ambiguous_ties"] = int(grp["label"].isna().sum())
grp = grp[grp["label"].notna()].copy()
grp["text"] = (grp["ticker"] + ": " + grp["title"]) if INPUT_MODE == "ticker_title" else grp["title"]
grp["label_id"] = grp["label"].map(LABEL2ID).astype(int)
grp["title_words"] = grp["title"].str.split().str.len()
cols = ["text", "title", "label", "label_id"] + [c for c in ("timestamp", "ticker") if c in grp] + \
       ["n_source_rows", "label_agreement", "had_conflict", "title_words"]
data = grp[cols].reset_index(drop=True)
report_clean["final_rows"] = len(data)
report_clean["final_label_distribution"] = data["label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()

if SMOKE_TEST:
    data = data.sample(n=min(SMOKE_TEST_ROWS, len(data)), random_state=SEED)
data = data.sort_values(["timestamp", "text"] if "timestamp" in data else ["text"], kind="stable").reset_index(drop=True)

out_clean = data.copy()
if "timestamp" in out_clean:
    out_clean["timestamp"] = out_clean["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
out_clean.to_csv(OUT / "cleaned_dataset.csv", index=False)
print("\n=== CLEANING ===")
for k, v in report_clean.items():
    print(f"{k}: {v}")
print(f"Saved {OUT / 'cleaned_dataset.csv'}")

# %% ==========================================================================
# 5. TRAIN / VALIDATION / TEST SPLIT (chronological)
# =============================================================================
# Earliest 70% -> train, next 15% -> validation, latest 15% -> test. No shuffling before the
# split: a random split would let the model see "future" news regimes during training.
n = len(data)
if "timestamp" in data:
    def cut(idx: int) -> int:     # never split rows that share the exact same timestamp
        while 0 < idx < n and data.at[idx, "timestamp"] == data.at[idx - 1, "timestamp"]:
            idx += 1
        return idx

    i_val, i_test = cut(int(n * TRAIN_FRAC)), cut(int(n * (TRAIN_FRAC + VAL_FRAC)))
    train_df, val_df, test_df = data.iloc[:i_val], data.iloc[i_val:i_test], data.iloc[i_test:]
    split_method = "chronological (time-based)"
else:
    from sklearn.model_selection import train_test_split

    train_df, tmp = train_test_split(data, test_size=VAL_FRAC + TEST_FRAC, stratify=data["label_id"], random_state=SEED)
    val_df, test_df = train_test_split(tmp, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
                                       stratify=tmp["label_id"], random_state=SEED)
    split_method = "stratified random (NO timestamp column - temporal leakage cannot be excluded)"
    print("WARNING:", split_method)
train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))
assert not set(train_df["text"]) & set(test_df["text"]), "leakage: identical text in train and test"
assert not set(train_df["text"]) & set(val_df["text"]), "leakage: identical text in train and validation"

split_info = {"method": split_method}
for name, d in (("train", train_df), ("validation", val_df), ("test", test_df)):
    info = {"rows": len(d), "label_distribution": d["label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()}
    if "timestamp" in d and len(d):
        info["date_range"] = [d["timestamp"].min().strftime("%Y-%m-%d %H:%M"), d["timestamp"].max().strftime("%Y-%m-%d %H:%M")]
    split_info[name] = info
    o = d.copy()
    if "timestamp" in o:
        o["timestamp"] = o["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    o.to_csv(OUT / f"{name}.csv", index=False)
print("\n=== SPLIT ===")
print(json.dumps(split_info, indent=2))

# %% ==========================================================================
# 6. CLASS IMBALANCE
# =============================================================================
# The classes are moderately imbalanced (neutral is the large majority). We use ONE technique:
# class-weighted cross-entropy (weights computed from the TRAINING split only).
#   * vs. weighted sampling: sampling duplicates minority titles and drops neutral ones per epoch,
#     which with ~150k texts wastes data and overfits repeated minority headlines; loss weighting
#     keeps every example and gives the same expected gradient balance.
#   * vs. focal loss: focal loss targets extreme imbalance / many easy negatives (e.g. 1:1000);
#     at roughly 6:1 it adds a hyper-parameter without a clear benefit.
# Model selection uses validation MACRO-F1 so minority classes count as much as neutral.
counts = train_df["label_id"].value_counts().reindex(range(len(LABELS))).fillna(0).to_numpy(dtype=float)
if CLASS_WEIGHTING == "balanced":
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
elif CLASS_WEIGHTING == "sqrt":
    weights = np.sqrt(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
else:
    weights = np.ones(len(LABELS))
CLASS_WEIGHTS = torch.tensor(weights, dtype=torch.float)
imbalance = {"train_counts": dict(zip(LABELS, counts.astype(int).tolist())),
             "method": f"class-weighted cross-entropy ({CLASS_WEIGHTING})",
             "class_weights": dict(zip(LABELS, np.round(weights, 4).tolist()))}
print("\n=== CLASS IMBALANCE ===")
print(json.dumps(imbalance, indent=2))

# %% ==========================================================================
# Metric helpers (shared by the baseline and FinBERT)
# =============================================================================
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_fscore_support)


def summary_metrics(y_true, y_pred) -> dict:
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f_m, "weighted_f1": f_w,
            "macro_precision": p_m, "macro_recall": r_m, "weighted_precision": p_w, "weighted_recall": r_w}


def per_class(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0)
    return {LABELS[i]: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} for i in range(len(LABELS))}


def rounded(d):
    if isinstance(d, dict):
        return {k: rounded(v) for k, v in d.items()}
    if isinstance(d, (float, np.floating)):
        return round(float(d), 4)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


# %% ==========================================================================
# 7. BASELINE: TF-IDF + LOGISTIC REGRESSION (benchmark only)
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

print("\n=== BASELINE: TF-IDF + Logistic Regression ===")
t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if SMOKE_TEST else 3, max_df=0.9, sublinear_tf=True,
                      max_features=300_000, strip_accents="unicode")
X_tr = vec.fit_transform(train_df["text"])          # fitted on TRAIN only
X_va, X_te = vec.transform(val_df["text"]), vec.transform(test_df["text"])
best_c, best_f1, best_lr = None, -1.0, None
for C in ([1.0] if SMOKE_TEST else [0.25, 1.0, 4.0]):   # C chosen on VALIDATION macro-F1
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED)
    lr.fit(X_tr, train_df["label_id"])
    f1v = f1_score(val_df["label_id"], lr.predict(X_va), average="macro")
    print(f"  C={C}: validation macro-F1 = {f1v:.4f}")
    if f1v > best_f1:
        best_c, best_f1, best_lr = C, f1v, lr
baseline_pred = best_lr.predict(X_te)                  # test used once, for the final comparison
baseline_metrics = summary_metrics(test_df["label_id"], baseline_pred)
baseline_info = {"C": best_c, "validation_macro_f1": best_f1, "vocabulary_size": len(vec.vocabulary_),
                 "fit_seconds": round(time.time() - t0, 1), "test": baseline_metrics,
                 "test_per_class": per_class(test_df["label_id"], baseline_pred)}
print("  test:", json.dumps(rounded(baseline_metrics)))
del X_tr, X_va, X_te

# %% ==========================================================================
# 8. FINBERT FINE-TUNING
# =============================================================================
import inspect  # noqa: E402

import torch.nn.functional as F  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,  # noqa: E402
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Token-length check -> is MAX_LENGTH sensible?
sample = train_df["text"].sample(min(20_000, len(train_df)), random_state=SEED).tolist()
lens = np.array([len(x) for x in tokenizer(sample, add_special_tokens=True)["input_ids"]])
token_stats = {"p50": int(np.percentile(lens, 50)), "p95": int(np.percentile(lens, 95)),
               "p99": int(np.percentile(lens, 99)), "max": int(lens.max()),
               "truncated_share_at_max_length": round(float((lens > MAX_LENGTH).mean()), 4)}
print("\nToken lengths:", token_stats)
if token_stats["truncated_share_at_max_length"] > 0.02:
    print(f"NOTE: {token_stats['truncated_share_at_max_length']:.1%} of titles exceed MAX_LENGTH={MAX_LENGTH}; "
          "consider raising it.")


def load_model():
    """FinBERT's 3-way head is (positive, negative, neutral). If REUSE_SENTIMENT_HEAD, its rows are
    re-ordered to (neutral, spike, crash) = (neutral, positive, negative) as a warm start; otherwise a
    fresh classification head is initialised."""
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    base_labels = {str(v).lower(): int(k) for k, v in base.config.id2label.items()}
    warm = {"neutral": "neutral", "spike": "positive", "crash": "negative"}
    if REUSE_SENTIMENT_HEAD and base.config.num_labels == len(LABELS) and all(warm.get(l) in base_labels for l in LABELS):
        order = [base_labels[warm[l]] for l in LABELS]
        with torch.no_grad():
            base.classifier.weight.copy_(base.classifier.weight[order].clone())
            base.classifier.bias.copy_(base.classifier.bias[order].clone())
        model, head = base, "reused FinBERT sentiment head (positive->spike, negative->crash, neutral->neutral)"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABELS), ignore_mismatched_sizes=True)
        head = "new randomly initialised classification head"
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.problem_type = "single_label_classification"
    return model, head


model, head_init = load_model()
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {MODEL_NAME} | head: {head_init} | parameters: {n_params:,} (trainable {n_trainable:,})")


def to_dataset(d: pd.DataFrame) -> Dataset:
    ds = Dataset.from_dict({"text": d["text"].tolist(), "labels": d["label_id"].tolist()})
    # truncation + attention masks here; padding happens per batch (dynamic padding) in the collator
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH),
                  batched=True, remove_columns=["text"])


train_ds, val_ds, test_ds = to_dataset(train_df), to_dataset(val_df), to_dataset(test_df)
collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if DEVICE.type == "cuda" else None)


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (see section 6)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels, weight=CLASS_WEIGHTS.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = summary_metrics(labels, preds)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"]}


steps_per_epoch = math.ceil(len(train_ds) / (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
total_steps = SMOKE_TEST_MAX_STEPS if SMOKE_TEST else math.ceil(steps_per_epoch * NUM_EPOCHS)
eval_steps = max(1, SMOKE_TEST_MAX_STEPS // 2 if SMOKE_TEST else steps_per_epoch // EVALS_PER_EPOCH)
use_fp16 = DEVICE.type == "cuda"      # mixed precision on Colab GPUs (T4/V100/A100)

ta_params = inspect.signature(TrainingArguments.__init__).parameters
args_kwargs = dict(
    output_dir=str(CKPT_DIR),
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    max_steps=SMOKE_TEST_MAX_STEPS if SMOKE_TEST else -1,
    weight_decay=WEIGHT_DECAY,                      # AdamW (default optimizer) with decoupled weight decay
    warmup_steps=int(WARMUP_RATIO * total_steps),   # linear warm-up, then linear decay
    lr_scheduler_type="linear",
    max_grad_norm=MAX_GRAD_NORM,                    # gradient clipping
    fp16=use_fp16,
    save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
    logging_steps=max(1, min(100, eval_steps // 4)),
    report_to="none",
    seed=SEED, data_seed=SEED,
    dataloader_num_workers=2 if DEVICE.type == "cuda" else 0,
    dataloader_pin_memory=DEVICE.type == "cuda",
)
# `evaluation_strategy` was renamed to `eval_strategy` in newer transformers versions
args_kwargs["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "steps"
args_kwargs["eval_steps"] = eval_steps
training_args = TrainingArguments(**args_kwargs)

trainer_kwargs = dict(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)])
trainer_kwargs["processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters
               else "tokenizer"] = tokenizer
trainer = WeightedTrainer(**trainer_kwargs)

last_ckpt = get_last_checkpoint(str(CKPT_DIR)) if RESUME_FROM_CHECKPOINT and not SMOKE_TEST else None
print(f"\n=== FINBERT TRAINING === steps/epoch={steps_per_epoch}, total steps={total_steps}, "
      f"eval every {eval_steps} steps, fp16={use_fp16}" + (f", resuming from {last_ckpt}" if last_ckpt else ""))
t0 = time.time()
train_result = trainer.train(resume_from_checkpoint=last_ckpt)
train_seconds = time.time() - t0
print(f"Training finished in {train_seconds / 60:.1f} min")

# %% ==========================================================================
# 9. TRAINING HISTORY
# =============================================================================
hist = trainer.state.log_history
train_logs = pd.DataFrame([h for h in hist if "loss" in h and "eval_loss" not in h])
eval_logs = pd.DataFrame([h for h in hist if "eval_loss" in h])
rows = []
for _, e in eval_logs.iterrows():
    prev = train_logs[train_logs["step"] <= e["step"]]
    rows.append({"step": int(e["step"]), "epoch": round(float(e["epoch"]), 3),
                 "train_loss": float(prev["loss"].tail(max(1, eval_steps // training_args.logging_steps)).mean())
                 if len(prev) else np.nan,
                 "validation_loss": e["eval_loss"], "validation_accuracy": e.get("eval_accuracy"),
                 "validation_macro_f1": e.get("eval_macro_f1"), "validation_weighted_f1": e.get("eval_weighted_f1")})
history = pd.DataFrame(rows)
history.to_csv(RESULTS / "training_history.csv", index=False)
if len(train_logs):
    train_logs[["step", "epoch", "loss"] + [c for c in ("learning_rate", "grad_norm") if c in train_logs]] \
        .to_csv(RESULTS / "training_loss_steps.csv", index=False)

best_row = history.loc[history["validation_macro_f1"].idxmax()] if len(history) else None


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=130)
    plt.close(fig)


if len(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if len(train_logs):
        axes[0].plot(train_logs["epoch"], train_logs["loss"], color="#9aa4ae", lw=1, label="training loss (per log step)")
    axes[0].plot(history["epoch"], history["train_loss"], "o-", color="#2f6fb0", label="training loss")
    axes[0].plot(history["epoch"], history["validation_loss"], "o-", color="#c4581c", label="validation loss")
    axes[0].set_title("Loss (class-weighted CE)")
    axes[1].plot(history["epoch"], history["validation_accuracy"], "o-", color="#2f6fb0")
    axes[1].set_title("Validation accuracy")
    axes[2].plot(history["epoch"], history["validation_macro_f1"], "o-", color="#2f6fb0", label="macro-F1")
    axes[2].plot(history["epoch"], history["validation_weighted_f1"], "o--", color="#9aa4ae", label="weighted-F1")
    axes[2].set_title("Validation F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    if best_row is not None:
        axes[2].axvline(best_row["epoch"], color="#59636e", ls=":", lw=1)
    _save(fig, "training_history.png")
    for col, title, name in (("train_loss", "Training loss", "training_loss.png"),
                             ("validation_loss", "Validation loss", "validation_loss.png"),
                             ("validation_accuracy", "Validation accuracy", "accuracy.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["epoch"], history[col], "o-", color="#2f6fb0")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        _save(fig, name)

# %% ==========================================================================
# 10. FINAL EVALUATION ON THE UNTOUCHED TEST SET
# =============================================================================
# The best checkpoint (by validation macro-F1) is already loaded. The test set was not used for
# training, model selection, early stopping or threshold choice - it is used exactly once here.
pred_out = trainer.predict(test_ds)
logits = pred_out.predictions
probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
y_true = test_df["label_id"].to_numpy()
y_pred = probs.argmax(axis=1)
confidence = probs.max(axis=1)

finbert_metrics = summary_metrics(y_true, y_pred)
finbert_per_class = per_class(y_true, y_pred)
report = classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=DISPLAY,
                               output_dict=True, zero_division=0)
pd.DataFrame(report).T.round(4).to_csv(RESULTS / "classification_report.csv")
print("\n=== TEST SET (FinBERT) ===")
print(classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=DISPLAY, digits=4,
                            zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
pd.DataFrame(cm, index=[f"actual_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS]).to_csv(
    RESULTS / "confusion_matrix.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mat, title, fmt in ((axes[0], cm, "Confusion matrix (counts)", "{:,}"),
                            (axes[1], cm / np.maximum(cm.sum(axis=1, keepdims=True), 1),
                             "Confusion matrix (row-normalised = recall)", "{:.1%}")):
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), DISPLAY)
    ax.set_yticks(range(len(LABELS)), DISPLAY)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    thr = mat.max() / 2
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > thr else "#1f2328")
_save(fig, "confusion_matrix.png")

# --- per-prediction output (confidence analysis) -------------------------------------------
pred_df = pd.DataFrame({"title": test_df["title"], "actual_label": test_df["label"],
                        "predicted_label": [LABELS[i] for i in y_pred], "confidence": confidence.round(6)})
for i, l in enumerate(LABELS):
    pred_df[f"prob_{l}"] = probs[:, i].round(6)
pred_df["correct"] = pred_df["actual_label"] == pred_df["predicted_label"]
if "ticker" in test_df:
    pred_df["ticker"] = test_df["ticker"]
if "timestamp" in test_df:
    pred_df["timestamp"] = test_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
pred_df.to_csv(RESULTS / "test_predictions.csv", index=False)

# --- error analysis ---------------------------------------------------------------------------
mis = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False)
mis_cols = ["title", "actual_label", "predicted_label", "confidence"] + [f"prob_{l}" for l in LABELS] + \
           [c for c in ("ticker", "timestamp") if c in mis]
mis[mis_cols].rename(columns={"confidence": "prediction_confidence"}).to_csv(
    RESULTS / "misclassified_examples.csv", index=False)
conf_pairs = (mis.groupby(["actual_label", "predicted_label"]).size().rename("count").reset_index()
              .sort_values("count", ascending=False))
conf_pairs["share_of_actual_class"] = (conf_pairs["count"] /
                                       conf_pairs["actual_label"].map(pred_df["actual_label"].value_counts())).round(4)
conf_pairs["share_of_all_errors"] = (conf_pairs["count"] / max(len(mis), 1)).round(4)
conf_pairs.to_csv(RESULTS / "confusion_summary.csv", index=False)
print("Most frequent confusions:\n", conf_pairs.to_string(index=False))

# --- calibration: does "confidence" match how often the model is right? -------------------------
bins = np.linspace(1 / len(LABELS), 1.0, 11)
bucket = np.clip(np.digitize(confidence, bins[1:-1]), 0, len(bins) - 2)
calib = []
ece = 0.0
for b in range(len(bins) - 1):
    m = bucket == b
    if m.any():
        acc, conf_mean = float((y_pred[m] == y_true[m]).mean()), float(confidence[m].mean())
        ece += m.mean() * abs(acc - conf_mean)
        calib.append({"confidence_bin": f"{bins[b]:.2f}-{bins[b + 1]:.2f}", "count": int(m.sum()),
                      "mean_confidence": round(conf_mean, 4), "accuracy": round(acc, 4)})
pd.DataFrame(calib).to_csv(RESULTS / "calibration.csv", index=False)
print(f"Expected calibration error (ECE): {ece:.4f}")

# --- baseline vs FinBERT --------------------------------------------------------------------
comparison = pd.DataFrame([
    {"model": "TF-IDF + Logistic Regression", **rounded(baseline_metrics)},
    {"model": f"Fine-tuned FinBERT ({MODEL_NAME})", **rounded(finbert_metrics)},
])
comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
print("\n=== TF-IDF Logistic Regression vs Fine-tuned FinBERT (test set) ===")
print(comparison.to_string(index=False))

# %% ==========================================================================
# 11. SAVE THE FINAL MODEL + LABEL MAPPING
# =============================================================================
trainer.save_model(str(MODEL_DIR))          # model weights + config (with id2label)
tokenizer.save_pretrained(str(MODEL_DIR))
label_mapping = {"id2label": {str(k): v for k, v in ID2LABEL.items()}, "label2id": LABEL2ID,
                 "input_mode": INPUT_MODE, "max_length": MAX_LENGTH, "base_model": MODEL_NAME,
                 "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "smoke_test": SMOKE_TEST}
(MODEL_DIR / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")
(RESULTS / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")
print(f"\nModel, tokenizer and label_mapping.json saved to {MODEL_DIR}")

# Sanity check: reload from disk and predict one example on CPU, exactly as predict.py will.
_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
_mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to("cpu").eval()
with torch.no_grad():
    _p = torch.softmax(_mdl(**_tok(test_df["text"].iloc[0], return_tensors="pt", truncation=True,
                                   max_length=MAX_LENGTH)).logits, dim=-1)[0].numpy()
assert np.allclose(_p, probs[0], atol=2e-3), "reloaded model gives different probabilities"
print("Reload check passed (saved model reproduces the test prediction on CPU).")

# %% ==========================================================================
# 12. METRICS.JSON + REPORT
# =============================================================================
metrics = rounded({
    **RUN_INFO,
    "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "data_path": DATA_PATH,
    "dataset_inspection": {k: v for k, v in inspection.items() if k != "columns_in_file"},
    "cleaning": report_clean,
    "input_mode": INPUT_MODE,
    "number_of_classes": len(LABELS),
    "label_mapping": ID2LABEL,
    "class_distribution_final": report_clean["final_label_distribution"],
    "split": split_info,
    "class_imbalance": imbalance,
    "token_length_stats": token_stats,
    "model": {"name": MODEL_NAME, "head_initialisation": head_init, "parameters": n_params,
              "trainable_parameters": n_trainable},
    "hyperparameters": {"max_length": MAX_LENGTH, "train_batch_size": TRAIN_BATCH_SIZE,
                        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS, "learning_rate": LEARNING_RATE,
                        "epochs": NUM_EPOCHS, "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
                        "max_grad_norm": MAX_GRAD_NORM, "fp16": use_fp16, "seed": SEED,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE, "eval_steps": eval_steps},
    "training_time_minutes": train_seconds / 60,
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_epoch": float(best_row["epoch"]) if best_row is not None else None,
    "best_validation_macro_f1": float(best_row["validation_macro_f1"]) if best_row is not None else None,
    "finbert_test": finbert_metrics,
    "finbert_test_per_class": finbert_per_class,
    "finbert_expected_calibration_error": ece,
    "baseline_tfidf_logreg": baseline_info,
    "test_majority_class_accuracy": float((y_true == np.bincount(train_df["label_id"]).argmax()).mean()),
})
(RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


def pct(x):
    return f"{100 * x:.2f}%"


fb, bl = metrics["finbert_test"], metrics["baseline_tfidf_logreg"]["test"]
lines = [
    "# FinBERT market-effect classifier - training report", "",
    "**Smoke test - numbers below are NOT meaningful.**" if SMOKE_TEST else "",
    f"- Generated: {metrics['finished_at_utc']}  |  device: {metrics['device']}  |  GPU: {metrics['gpu']}",
    f"- Model: `{MODEL_NAME}` ({n_trainable:,} trainable parameters), head: {head_init}",
    f"- Training time: {train_seconds / 60:.1f} min; best checkpoint at epoch {metrics['best_epoch']} "
    f"(validation macro-F1 {metrics['best_validation_macro_f1']})", "",
    "## Data",
    f"- Source rows: {inspection['rows']:,}; unique model inputs after cleaning/de-duplication: {len(data):,}",
    f"- Classes ({len(LABELS)}): " + ", ".join(f"{i} = {l}" for i, l in ID2LABEL.items()),
    f"- Final class distribution: {report_clean['final_label_distribution']}",
    f"- Headlines with conflicting labels across tickers/dates: {report_clean['inputs_with_conflicting_labels']:,} "
    f"(resolved by majority vote; {report_clean['dropped_ambiguous_ties']:,} exact ties dropped)",
    f"- Split: {split_method}",
] + [f"  - {k}: {v['rows']:,} rows, {v.get('date_range', '')}, {v['label_distribution']}"
     for k, v in split_info.items() if k != "method"] + [
    f"- Class imbalance handling: {imbalance['method']}, weights {imbalance['class_weights']}", "",
    "## Test results (untouched test set)", "",
    "| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro precision | Macro recall |",
    "|---|---|---|---|---|---|",
    f"| TF-IDF + Logistic Regression | {pct(bl['accuracy'])} | {bl['macro_f1']:.4f} | {bl['weighted_f1']:.4f} | "
    f"{bl['macro_precision']:.4f} | {bl['macro_recall']:.4f} |",
    f"| Fine-tuned FinBERT | {pct(fb['accuracy'])} | {fb['macro_f1']:.4f} | {fb['weighted_f1']:.4f} | "
    f"{fb['macro_precision']:.4f} | {fb['macro_recall']:.4f} |", "",
    f"Always predicting the majority class would score {pct(metrics['test_majority_class_accuracy'])} accuracy, "
    "which is why macro-F1 (all classes weighted equally) is the main metric.",
    f"Expected calibration error of FinBERT confidences: {ece:.4f} (see calibration.csv).", "",
    "## Data-leakage prevention",
    "- **Model input = the news title only.** No price, return, volume, OHLCV, volatility, sentiment or label-",
    "  generation field is given to the model. Post-news market movement was used by the data pipeline solely",
    "  to create the target label (spike / crash / neutral).",
    "- Chronological split: the model is trained on older news and tested on newer news, so it cannot learn from",
    "  the future. Rows with the same timestamp are never split across sets.",
    "- Each distinct headline appears in exactly one split (asserted in code).",
    "- TF-IDF vocabulary and class weights are fitted on the training split only; hyper-parameters, early stopping",
    "  and the best checkpoint are chosen on validation; the test set is used once, at the end.", "",
    "## Interpretation",
    "- The model predicts the dataset's *historical label definition*: whether the stock showed a statistically",
    "  unusual reaction after similar headlines. It does **not** establish that a headline *causes* a price move.",
    "- The confidence is the softmax probability of the predicted class. It is a model score, not a guaranteed",
    "  probability that the market will move; check calibration.csv for how well it matches observed accuracy.",
]
(RESULTS / "report.md").write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")
print(f"\nAll results written to {RESULTS}")
print("Files:", sorted(p.name for p in RESULTS.iterdir()))
