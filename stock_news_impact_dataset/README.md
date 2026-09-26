# Stock News Impact Dataset

A reproducible pipeline that builds a large, clean dataset of **US stock-market news events**, each linked to the
affected ticker(s), aligned to **Alpaca 1-minute OHLCV bars**, and labelled `spike` / `crash` / `neutral` according to
the market reaction that followed.

> **Scope of this phase:** COLLECT → CLEAN → DEDUPLICATE → NER/TICKER → MARKET DATA → REACTION FEATURES → LABEL →
> FINAL DATASET. No model training happens here.

> **Causality disclaimer.** A market reaction that follows a news article is an **association, not proof** that the
> article caused the price movement. A label means only that *the stock showed a statistically unusual market
> reaction after this news event*. Other news, macro events, order flow or coincidence may be responsible.

---

## Repository contents

| Path | What it is |
|---|---|
| `main.py`, `collectors/`, `nlp/`, `labeling/`, `processing/`, `storage/`, `config/`, `tests/` | Data-collection and labelling pipeline |
| `data/processed/ml_ready_dataset.csv` (Git LFS) | Final training dataset: 390,057 labelled news events, pre-news features + target only |
| `data/processed/*.json`, `data_quality_report.csv`, `duplicate_map.csv`, `rejected_news.csv`, `plots/` | Quality report, schema, de-duplication audit and inspection plots |
| `finbert_market_effect/` | FinBERT training (`train_finbert.py`, Colab), local inference (`predict.py`) and its own README |
| `finbert_market_effect/model/finbert_market_effect/` (Git LFS) | Fine-tuned FinBERT model, tokenizer and `label_mapping.json` |

Not included, because of size or because they can be regenerated:
- the raw API downloads (`data/raw/`, including about 429k one-minute bar files)
- the intermediate CSVs over 100 MB
- checkpoints, logs and virtual environments

`git lfs install` must be run before cloning, otherwise the two LFS files download as small pointer files.

**To rebuild everything from scratch:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
copy .env.example .env      # then add your API keys
python main.py
```

## 1. Project purpose

The goal is an academic ML dataset for news-impact prediction: given the text and metadata of a news article, predict
whether the mentioned stock will show an unusual upward move (`spike`), an unusual downward move (`crash`), or
nothing unusual (`neutral`). The target size is 40,000-60,000+ usable ticker-events. **Nothing is fabricated.** If the
APIs return less under your subscription, the pipeline reports the real count (see `data_quality_report.json`,
field `usable_labelled_events`).

## 2. API setup

| Service | What you need | Where |
|---|---|---|
| Alpha Vantage | API key | https://www.alphavantage.co/support/#api-key |
| Alpaca | API key ID + secret (paper or live account both work) | https://app.alpaca.markets → API Keys |

- **Alpha Vantage free keys allow 25 requests/day.** One request returns up to 1,000 articles, so a two-year range
  needs hundreds to thousands of requests. With a free key the collector stops cleanly when the daily quota is used
  and continues the next day from the same point. A premium key is strongly recommended. Set `AV_REQUESTS_PER_MINUTE`
  to match it and `AV_DAILY_REQUEST_LIMIT=0`.
- **Alpaca Basic (free)** allows 200 requests/min, historical news, and historical SIP bars older than 15 minutes.
  If SIP is not permitted for your account, the pipeline switches to the IEX feed automatically and records this in
  the `feed` column of every cached bar file.

## 3. `.env` configuration

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Fill in `ALPHA_VANTAGE_API_KEY`, `ALPACA_API_KEY`, and `ALPACA_SECRET_KEY`. Everything else has sensible defaults.
`.env.example` documents every option. The most important ones are:

| Variable | Default | Meaning |
|---|---|---|
| `START_DATE` / `END_DATE` | 2024-09-24 / 2026-09-25 | Inclusive UTC date range |
| `MAX_WORKERS` | 8 | Size of the single, flat thread pool (Alpaca only) |
| `ALPACA_PAPER` | true | `false` if you use live-account keys (affects the calendar/asset endpoints) |
| `ALPHA_VANTAGE_TIMEZONE` | auto | Time zone of Alpha Vantage's naive timestamps (see §11) |
| `ALPACA_NEWS_SYMBOLS` | *(all)* | Optionally restrict Alpaca news to some symbols |
| `SPIKE_RETURN_THRESHOLD` etc. | see §12 | Labelling parameters |

Keys are read only from `.env` via `python-dotenv`. They are never hard-coded, logged, or written to any output.
Failed requests are logged with the `apikey` parameter masked. `.env` is listed in `.gitignore`.

## 4. Installation

Python 3.10+ is required (tested on 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py --check-api       # verifies both APIs from .env (uses 1 Alpha Vantage request)
```

If the spaCy model cannot be installed, NER still runs using API symbols, cashtags, exchange tags and the alias
table. The log warns that entity checking is disabled.

## 5. Running the collector

```bash
python main.py              # complete pipeline
python main.py --news       # collect Alpha Vantage + Alpaca news, then clean + de-duplicate
python main.py --dedup      # re-run cleaning + de-duplication only
python main.py --ner        # ticker identification
python main.py --market     # Alpaca 1-minute bars, alignment, reaction features
python main.py --label      # spike / crash / neutral
python main.py --build      # final CSV + quality report
python main.py --inspect    # printed summary + plots
```

Other flags:

- `--skip-alpha-vantage` / `--skip-alpaca-news` skip one news source for this run.
- `--force` together with a stage (for example `--ner --force` or `--market --force`) redoes that stage. Old outputs are
  **renamed to `*.archived.*`, never deleted**. `--market --force` recomputes features from the cached bars without
  making any API calls.
- `--env path/to/file.env` uses a different config file.

Stages can be run in any order. Each one uses whatever the previous stages have produced so far. A practical workflow
with a free Alpha Vantage key is to run `python main.py` once per day. Alpaca news, market data, NER and labels
advance on every run, while Alpha Vantage adds 25 requests' worth of news per day.

## 6. Resume / checkpoint behaviour

- Only one `python main.py` can run on a data folder at a time (an OS file lock); a second one exits immediately.
- **Ctrl+C once** makes in-flight requests finish, saves checkpoints, and exits with code 130. **Ctrl+C twice**
  forces an immediate exit. Committed work is still safe (see below).
- Rerunning the **same command** continues where it stopped.

| Stage | Checkpoint | Resume granularity |
|---|---|---|
| Alpha Vantage news | `data/checkpoints/alpha_vantage_news.json` (window queue + cursor) | per request |
| Alpaca news | `data/checkpoints/alpaca_news.json` (chunk → `next_page_token`) | per page (50 articles) |
| Alpaca market data | one cache file per ticker-day in `data/raw/market/` | per ticker-day |
| Market features | `data/checkpoints/market_features.json` + event ids already written | per 2,000 events |
| NER | `data/checkpoints/ner.json` + `ner_processed.csv` | per chunk (5,000 articles) |
| Labelling | `data/checkpoints/labeling.json` (config hash + input signature) | skipped when nothing changed |

Crash safety works like this:

- Rows are **appended incrementally** and fsync'ed.
- Each checkpoint stores the "committed" size of its CSV and is written atomically (temp file → `os.replace`, with a
  `.bak` copy).
- If the process is killed between writing rows and saving the checkpoint, the next run moves the uncommitted tail to
  an `*.orphan` file and re-collects those rows. Raw files are never truncated beyond the last checkpoint and never
  deleted.
- Raw rows are also de-duplicated by provider id on resume.

## 7. Data sources

| File | Content |
|---|---|
| `data/raw/alpha_vantage_news.csv` | Alpha Vantage NEWS_SENTIMENT records, all original fields (authors, topics, per-ticker sentiment JSON, `time_published` string as returned) |
| `data/raw/alpaca_news.csv` | Alpaca `/v1beta1/news` records (id, headline, summary, author, created/updated time, symbols, source, url, images) |
| `data/raw/raw_news.csv` | Both providers in one normalised schema (pre-dedup) |
| `data/raw/market/{TICKER}_{YYYY-MM-DD}.csv` | Alpaca **1-minute** bars for one regular session, exactly as returned |
| `data/raw/reference/us_equities.csv` | Alpaca asset master used to validate tickers |
| `data/raw/reference/trading_calendar.csv` | Alpaca trading calendar (holidays, early closes) |

Every news row has `news_provider` (`alpha_vantage` or `alpaca`, meaning *which API delivered it*) and
`source`/`publisher` (*who wrote it*, for example `news_provider=alpaca, source=benzinga, publisher=Benzinga`).

## 8. Alpha Vantage role

- Provides news plus **text-derived sentiment**: `overall_sentiment_score` and per-ticker sentiment/relevance.
- The endpoint has no page token. The collector walks the date range with a **time cursor**. Each request uses
  `sort=EARLIEST&limit=1000`. When a response is full, the next request starts at the minute of the last article
  returned, so saturated windows are fully paginated.
- It handles Alpha Vantage's HTTP-200 "Information/Note" messages. Per-minute limits → wait 65 s and retry. Daily
  quota → save and stop this source; the other stages still run.
- Alpha Vantage returns no article id, so `alpha_vantage_id` is a SHA-1 of the article URL.

## 9. Alpaca role

Alpaca is used for **two things**:

1. **News:** Benzinga-sourced articles with editor-tagged `symbols`, paginated with `next_page_token` in parallel
   weekly chunks.
2. **Market data:** **1-minute bars only** (`timeframe=1Min`), plus the trading calendar and asset list.

No daily or other aggregated bars are requested. Volatility and volume baselines are also computed from 1-minute bars.

The request strategy works as follows:

- Events are reduced to unique `(ticker, session date)` pairs: the aligned session plus
  `BASELINE_LOOKBACK_SESSIONS` prior sessions.
- Pairs are grouped by date and fetched as **multi-symbol requests** (`ALPACA_SYMBOLS_PER_REQUEST`, paginated).
- Results are saved as one cache file per ticker-day and reused by every event that needs them.
- A day with genuinely no trades gets a header-only file, so it is not re-requested. A day whose request *failed*
  gets no file and is retried on the next run.
- Every failure is recorded in `data/failed_requests.csv`.

## 10. NER / ticker extraction (`nlp/ticker_ner.py`)

Evidence is combined with a noisy-OR. Independent agreeing sources raise the confidence.

| Method | Example | Confidence |
|---|---|---|
| `api_alpaca` | Alpaca `symbols` field | 0.85 |
| `api_alpha_vantage` | AV `ticker_sentiment`, scaled by relevance | 0.45–0.95 |
| `exchange_tag` | "(NASDAQ: AAPL)" | 0.95 |
| `cashtag` | "$TSLA" | 0.90 |
| `paren_ticker` | "Apple (AAPL)" (common acronyms such as AI, CEO or EV are excluded) | 0.80 |
| `ner_alias` | "Nvidia", "Advanced Micro Devices", "Home Depot" | 0.60–0.85 |

- Aliases come from a curated table (Apple→AAPL, Google→GOOGL, Facebook→META, ...) and from the cleaned legal names
  in Alpaca's asset master.
- **spaCy** (`en_core_web_sm`) provides entity spans. Ambiguous single-word names (Target, Block, Snap, Visa,
  Shell, ...) are accepted only when spaCy tags them as an ORG, so "price target" is not treated as TGT.
- **Upper-case words are never treated as tickers on their own.**
- Every symbol must be in the US-equity reference list: active, NYSE/NASDAQ/AMEX/ARCA/BATS listed, and not an ETF,
  fund, warrant, unit, right or preferred share, unless `INCLUDE_ETFS=true`. Crypto (`CRYPTO:BTC`) and forex
  (`FOREX:USD`) tags are discarded.
- Multi-ticker articles produce one row per ticker (`event_id = news_id + "_" + ticker`) with a shared `news_id`.
  Nothing is lost.
- Output columns: `ticker`, `company_name`, `ticker_confidence`, `ticker_detection_method`, `n_tickers_in_article`.

## 11. Market-data alignment

All times are UTC internally. Bar timestamps are the **start** of each 1-minute bar.

| `market_alignment` | When |
|---|---|
| `same_minute` | News inside the regular session and a bar exists for that minute. News at 14:32:17 → bar 14:32:00–14:32:59. |
| `next_candle` | News inside the session, no trade that minute → first later bar (within `MAX_ALIGNMENT_GAP_MINUTES`) |
| `next_market_open` | Pre-market, after-hours, weekend or holiday news → first bar of the next regular session |
| `unavailable` | No usable bar. OHLCV stays **empty**, never invented; the reason is in `market_unavailable_reason`. |

Window prices (T+1, T+5, T+15, T+30, T+60) are the close of the last *actual* bar at or before the window end, which
is the last traded price. Windows that run past the session close are left empty. Session times, including early
closes, come from the trading calendar.

**Alpha Vantage timestamps** are naive strings (for example `20240924T143217`) and are stored untouched in
`published_timestamp_raw`. With `ALPHA_VANTAGE_TIMEZONE=auto`, the cleaner compares them with Alpaca's explicit-UTC
timestamps **for the same article URL** and picks the zone (UTC or America/New_York) that makes them agree. The
evidence is written to `data/processed/alpha_vantage_timezone.json`. With fewer than 20 overlapping articles it
falls back to UTC and says so; set the variable explicitly if you know the zone. Timestamps are never shifted by a
guess or randomly altered.

## 12. Label-generation methodology (`labeling/impact_labeler.py`)

The labeler uses these per-event quantities:

- `return_h = (close[T0+h] − close[T0]) / close[T0]` for h ∈ `LABEL_HORIZONS` (5, 15, 30, 60 min)
- `volatility` (σ): pre-news standard deviation of per-minute log returns. It uses `BASELINE_LOOKBACK_SESSIONS`
  prior sessions plus same-day bars before the news, with gaps between trades scaled by √Δt.
- `abnormal_return_h = return_h / (σ·√h)`: a z-score relative to *this* stock's normal noise
- `volume_ratio`: event-window volume (`VOLUME_WINDOW_MINUTES`) divided by the volume in the **same clock-time
  window** of prior sessions. This controls for the intraday U-shape, for example the heavy volume at the open.
- For overnight news: `gap_return` (open vs. last pre-news close) with z = gap/(σ·√390)

A horizon qualifies as `spike`/`crash` only if **all** of these hold:

1. `|z| ≥ ABNORMAL_RETURN_Z` (statistically unusual for this stock)
2. `|return| ≥ MIN_ABS_RETURN` (economically non-trivial)
3. Confirmation: `volume_ratio ≥ VOLUME_SPIKE_RATIO`, **or** the return exceeds
   `SPIKE_RETURN_THRESHOLD` / `CRASH_RETURN_THRESHOLD`. Set `REQUIRE_VOLUME_CONFIRMATION=true` to always require the
   volume signal.
4. Direction consistency across horizons ≥ `DIRECTION_CONSISTENCY_MIN`. For overnight news this uses the cumulative
   move from the last pre-news close.

The decision rules are:

- If qualifying horizons exist in one direction only, the label is `spike` or `crash` from the strongest one.
- If both directions qualify, the reaction is conflicting and the label is `neutral`.
- Otherwise the label is `neutral`.
- Events without aligned market data or with fewer than `MIN_BASELINE_BARS` baseline returns get **no label**.
  They are flagged, never guessed.

Supporting columns:

- `label_confidence` (0.5–1) is a transparent heuristic. It is a weighted blend of how far |z| exceeds the
  threshold, volume strength, return size vs. threshold, and direction consistency. It is **not** a calibrated
  probability.
- `label_reason` explains every decision, for example: *"Strong positive 5-minute abnormal return (+3.13%, z=+27.7)
  with 5.9x normal volume; direction consistent in 100% of horizons"*.
- `label_horizon_minutes` and `abnormal_return` record which horizon decided the label.

## 13. Features vs target (anti-leakage)

`data/processed/dataset_schema.json` gives every column a role. **Only use `feature` columns as model inputs.**

**FEATURES** are known at publication time:

- news text: `title`, `description`
- `ticker`, `company_name`, `publisher`, `ticker_confidence`, `n_tickers_in_article`
- timestamp-derived: `news_published_timestamp_utc/et`, `news_date`, `news_time`, `news_day_of_week`,
  `news_hour_et`, `news_session_position`
- sentiment (text-derived by Alpha Vantage): `overall_sentiment_*`, `av_ticker_*`, `topics`
- **pre-news** market context: `volatility`, `baseline_volume_per_min`, `baseline_return_count`, `pre_news_close`,
  `pre_news_return_15m`

**TARGET:** `label` (spike / crash / neutral)

**OUTCOMES** are future data used *only* to generate the label. **Never feed these to a model.**

- the T0 candle `open`, `high`, `low`, `close`, `volume` (the T0 candle overlaps or follows publication)
- `return_1m`, `return_5m`, `return_15m`, `return_30m`, `return_60m`
- `high_move_5m`, `low_move_5m`, `bars_in_5m`
- `event_volume`, `expected_volume`, `volume_ratio`
- `gap_return`, `gap_abnormal`
- `abnormal_return_*`, `abnormal_return`
- `label_confidence`, `label_reason`, `label_horizon_minutes`, `direction_consistency`: these are derived from the
  outcomes. `label_confidence` may be used as a sample weight, never as an input.

For convenience, `data/processed/ml_ready_dataset.csv` contains only usable rows and only feature + target columns.
By default it also drops `news_id`, `event_id`, `publisher`, `alpha_vantage_id` and `alpaca_id` (configurable with
`ML_EXCLUDE_COLUMNS` in `.env`; the final dataset keeps them). Outcome columns are removed so they cannot leak into training by accident.

Use a **time-based train/test split** (for example by `news_date`). A random split lets the same day's market regime
leak between the sets.

## 14. Dataset schema (`final_stock_news_impact_dataset.csv`)

One row per (article, ticker).

| Group | Columns |
|---|---|
| ids | `news_id`, `event_id`, `alpha_vantage_id`, `alpaca_id` |
| text | `title`, `description` |
| ticker | `ticker`, `company_name`, `ticker_confidence`, `ticker_detection_method`, `n_tickers_in_article` |
| time | `news_published_timestamp_utc`, `news_published_timestamp_et`, `news_date`, `news_time`, `news_day_of_week`, `news_hour_et`, `news_session_position` |
| provenance | `news_provider`, `news_providers_all`, `publisher`, `source`, `source_domain`, `author`, `url` |
| sentiment | `overall_sentiment_score`, `overall_sentiment_label`, `av_ticker_relevance_score`, `av_ticker_sentiment_score`, `av_ticker_sentiment_label`, `topics` |
| pre-news market | `volatility`, `baseline_volume_per_min`, `baseline_return_count`, `pre_news_close`, `pre_news_return_15m` |
| alignment | `market_alignment`, `market_unavailable_reason`, `market_session_date`, `market_timestamp_utc`, `alignment_lag_seconds` |
| T0 candle *(outcome)* | `open`, `high`, `low`, `close`, `volume` |
| reaction *(outcome)* | `return_{1,5,15,30,60}m`, `high_move_5m`, `low_move_5m`, `bars_in_5m`, `event_volume`, `expected_volume`, `volume_ratio`, `gap_return`, `gap_abnormal`, `abnormal_return_{..}m`, `abnormal_return` |
| label | `label`, `label_confidence`, `label_reason`, `label_horizon_minutes`, `direction_consistency` |
| quality | `quality_flags`, `is_usable` |

Intermediate files in `data/processed/`:

| File | Content |
|---|---|
| `deduplicated_news.csv` | Unique articles |
| `duplicate_map.csv` | Every merge and the rule that caused it |
| `rejected_news.csv` | Invalid records with `reject_reason` |
| `stock_news_dataset.csv` | Ticker events after NER (`NEWS_OUTPUT`) |
| `market_data.csv` | Aligned market features (`MARKET_OUTPUT`) |
| `labeled_events.csv` | Market features + labels |

## 15. Data quality checks

Records are **flagged, not silently deleted**:

- Invalid news goes to `rejected_news.csv` with one of: `missing_title`, `missing_timestamp`,
  `malformed_timestamp`, `outside_date_range`.
- Final rows carry `quality_flags`: `invalid_ticker`, `missing_market_data`, `missing_market_data_pending`,
  `impossible_ohlc`, `negative_volume`, `unlabelled`, and others.
- Duplicate events are removed and counted. `is_usable` is True only for labelled rows with no critical flag.

De-duplication (`processing/deduplicator.py`) applies these rules in order:

1. normalised URL
2. normalised title + timestamp (±10 min)
3. normalised title + publisher (±6 h, titles ≥ 5 words)
4. fuzzy title ≥ 95 (same publisher, ±2 h, titles ≥ 6 words)

Rules 2–4 **never merge two records from the same provider with different URLs**. Those are distinct articles with
similar headlines, such as daily "stocks moving pre-market" posts.

`data_quality_report.json` / `.csv` contain:

- `raw_news_count`, `deduplicated_news_count`, `duplicate_count`
- `ticker_events`, `market_matched_events`, `missing_market_data`
- `spike_count`, `crash_count`, `neutral_count`, `invalid_count`
- the alignment breakdown, flag counts, `usable_labelled_events`, and whether the 40k target was reached

The de-duplication statistics are also printed after `--news`/`--dedup`.

## 16. API rate-limit considerations

- A sliding-window limiter is shared by **all** Alpaca calls (`ALPACA_REQUESTS_PER_MINUTE`, default 180 < 200).
  Alpha Vantage has its own limiter and an optional local daily cap.
- Retries use exponential backoff with jitter. They cover HTTP 408/425/429/5xx, timeouts and connection errors and
  honour `Retry-After`. Other 4xx errors fail fast and are logged to `failed_requests.csv`.
- There is one flat `ThreadPoolExecutor` per stage (`MAX_WORKERS`) and no nested pools. Alpha Vantage is sequential.
- Rough request budget for two years: Alpaca news ≈ articles/50 pages, and bars ≈ (ticker-days)/50 requests (plus
  pages). Tens of thousands of requests take a few hours at 180/min.

## 17. Loading the final CSV in Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd, json
df = pd.read_csv("/content/drive/MyDrive/stock_news_impact_dataset/data/processed/final_stock_news_impact_dataset.csv")
df = df[df["is_usable"]]                                   # labelled rows that passed all checks

schema = json.load(open(".../data/processed/dataset_schema.json"))
features = [c for c, m in schema.items() if m["role"] == "feature"]
X, y = df[features], df["label"]                           # outcome columns are excluded by construction
```

Or load `ml_ready_dataset.csv`, which already contains only ids, features and the target. Text fields are standard
RFC-4180 quoted CSV (commas, quotes and newlines escaped), UTF-8. They load directly in pandas, scikit-learn,
PyTorch/TensorFlow data pipelines, and Hugging Face `datasets.load_dataset("csv", data_files=...)`.

## 18. Limitations

- **Association ≠ causation.** Labels describe the reaction that followed an article, not its effect.
  Simultaneous news, earnings releases and market-wide moves all fall inside the windows. No market-index
  adjustment is applied, because this project uses 1-minute single-stock bars only.
- **Timestamp precision.** Publication time is the provider's. Articles can be updated later, and syndicated copies
  may lag the original. The T0 candle can begin up to 59 s before the news, which is why it is an outcome, not a
  feature.
- **Alpha Vantage free quota** (25 requests/day) makes full two-year coverage slow; see §2. Alpha Vantage's
  timestamp zone is inferred (§11).
- **Coverage depends on the providers.** Alpaca news is predominantly Benzinga. The final count depends on your
  subscription and is reported, never padded.
- **Ticker universe:** currently active US listings. Tickers delisted since publication are dropped. Alias matching can still miss unusual company references or rare false positives, and
  `ticker_confidence` and `ticker_detection_method` let you filter.
- **Regular session only.** Pre/after-hours reactions are measured through the opening gap (`gap_return`) at the
  next open, not tick by tick.
- **Volatility baseline** uses whole-session per-minute σ, without intraday-seasonality adjustment of σ (volume is
  seasonality-adjusted). Opening-minute moves are therefore somewhat more likely to look abnormal.
- **`label_confidence`** is a heuristic score, not a probability.
- Choosing the strongest of several horizons slightly raises the chance of a false `spike`/`crash` (multiple
  testing). Raise `ABNORMAL_RETURN_Z` for a stricter dataset.

## Tests

```bash
python -m pytest -q
```

The tests run the **entire pipeline offline** against a fake HTTP layer (`tests/fake_apis.py`) that mimics the real
API response formats. The synthetic fixtures exist only inside pytest's temp directories and are never used by
`main.py`. The tests cover:

- both raw news files and provider/publisher separation
- cross-provider de-duplication, and that similar same-provider articles are kept apart
- ticker extraction: ETF, crypto and acronym exclusion, and multi-ticker splitting
- `same_minute`, `next_candle`, `next_market_open` and `unavailable` alignment
- spike, crash and neutral labels, the final schema, and the quality report
- Ctrl+C resume from a page token, truncation of half-written rows, and no re-download on rerun
- no secrets in any output
