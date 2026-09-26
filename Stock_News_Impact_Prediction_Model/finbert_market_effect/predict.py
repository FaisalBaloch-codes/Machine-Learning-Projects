# -*- coding: utf-8 -*-
"""
Local (CPU) inference for the fine-tuned FinBERT market-effect model.

    python predict.py                                   # interactive: asks for a title
    python predict.py "Apple reports record quarterly revenue"
    python predict.py --model-dir path/to/finbert_market_effect "headline 1" "headline 2"

Reusable from other code (e.g. a Flask app):

    from predict import predict_news_effect
    predict_news_effect("Apple reports record quarterly revenue")
    # {'prediction': 'spike', 'confidence': 0.87, 'probabilities': {'neutral': ..., 'spike': ..., 'crash': ...}}

The confidence is the softmax probability of the predicted class - a model score, not a
guarantee about the market. Predictions reflect the historical label definition of the
training data and do not imply that a headline causes a price movement.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model" / "finbert_market_effect"
DEVICE = "cpu"   # inference always runs on CPU; no GPU required

# ---------------------------------------------------------------- text cleaning
# Must stay identical to clean_title() in train_finbert.py.
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f​-‏  ﻿]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                         "“": '"', "”": '"', "„": '"', "″": '"',
                         "–": "-", "—": "-", "−": "-", " ": " "})


def clean_title(text) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = html.unescape(str(text))
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    s = _CTRL.sub(" ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- model loading
@lru_cache(maxsize=4)
def load_model(model_dir: str | os.PathLike | None = None):
    """Load tokenizer, model and label mapping once (cached)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()          # keep the console clean (no download/loading bars)
    hf_logging.disable_progress_bar()
    path = Path(model_dir or os.environ.get("FINBERT_MODEL_DIR") or DEFAULT_MODEL_DIR)
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"No trained model found in '{path}'. Copy the 'finbert_market_effect' folder produced by "
            "train_finbert.py into model/ (or pass --model-dir / set FINBERT_MODEL_DIR).")
    mapping_file = path / "label_mapping.json"
    mapping = json.loads(mapping_file.read_text(encoding="utf-8")) if mapping_file.exists() else {}
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(DEVICE).eval()
    id2label = {int(k): v for k, v in (mapping.get("id2label") or model.config.id2label).items()}
    torch.set_grad_enabled(False)
    return tokenizer, model, id2label, int(mapping.get("max_length", 64)), mapping.get("input_mode", "title")


def predict_batch(titles: list[str], tickers: list[str | None] | None = None,
                  model_dir: str | None = None) -> list[dict]:
    """Predict several titles at once. Returns one result dict per title."""
    import torch

    tokenizer, model, id2label, max_length, input_mode = load_model(model_dir)
    cleaned = [clean_title(t) for t in titles]
    if any(not c for c in cleaned):
        raise ValueError("Empty title - please provide a news headline.")
    if input_mode == "ticker_title":
        if not tickers or any(not t for t in tickers):
            raise ValueError("This model was trained with INPUT_MODE='ticker_title'; a ticker is required.")
        cleaned = [f"{t.upper()}: {c}" for t, c in zip(tickers, cleaned)]
    enc = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
    results = []
    for row in probs:
        best = int(row.argmax())
        results.append({
            "prediction": id2label[best],
            "confidence": round(float(row[best]), 4),
            "probabilities": {id2label[i]: round(float(p), 4) for i, p in enumerate(row)},
        })
    return results


def predict_news_effect(title: str, ticker: str | None = None, model_dir: str | None = None) -> dict:
    """Predict the market effect of one news title.

    >>> predict_news_effect("Apple reports record quarterly revenue")
    {'prediction': 'spike', 'confidence': 0.8742, 'probabilities': {'neutral': 0.0521, 'spike': 0.8742, 'crash': 0.0737}}
    """
    return predict_batch([title], [ticker] if ticker else None, model_dir)[0]


def format_result(title: str, result: dict) -> str:
    line = "=" * 40
    probs = "\n".join(f"{k.capitalize()}: {v * 100:.2f}%" for k, v in result["probabilities"].items())
    return (f"{line}\nNEWS MARKET EFFECT PREDICTION\n{line}\n\nTitle:\n{title}\n\n"
            f"Prediction:\n{result['prediction'].upper()}\n\nConfidence:\n{result['confidence'] * 100:.2f}%\n\n"
            f"Probabilities:\n{probs}\n{line}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict the market effect of a news title (CPU).")
    parser.add_argument("titles", nargs="*", help="news title(s); omit for interactive mode")
    parser.add_argument("--model-dir", default=None, help=f"model folder (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--ticker", default=None, help="only for models trained with INPUT_MODE='ticker_title'")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of the formatted box")
    args = parser.parse_args()

    try:
        print("Loading model (CPU)...", file=sys.stderr)
        load_model(args.model_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    def show(title: str) -> None:
        try:
            res = predict_news_effect(title, args.ticker, args.model_dir)
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        print(json.dumps(res, indent=2) if args.json else format_result(clean_title(title), res))

    if args.titles:
        for t in args.titles:
            show(t)
        return 0
    print("Type a news headline and press Enter (empty line or 'quit' to exit).")
    while True:
        try:
            title = input("\nEnter news title: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not title or title.lower() in {"quit", "exit", "q"}:
            break
        show(title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
