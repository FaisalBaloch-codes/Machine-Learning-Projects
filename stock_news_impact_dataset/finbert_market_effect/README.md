# FinBERT News → Market-Effect Classifier

This project fine-tunes **FinBERT (`ProsusAI/finbert`)** to predict, from a news **headline alone**, which market
reaction the stock showed after similar news:

```text
0 = neutral     1 = spike     2 = crash
```

```text
Input : "Apple reports record quarterly revenue"
Output: Predicted Market Effect: SPIKE   Confidence: 87.42%   (illustrative output format, not a real result)
```

It has two parts:

- **Training** runs in Google Colab (`train_finbert.py`): inspect → clean → time-based split → TF-IDF baseline →
  FinBERT fine-tuning → evaluation and reports → saved model.
- **Prediction** runs locally on CPU in VS Code (`predict.py`): headline → fine-tuned FinBERT → label, confidence
  and class probabilities.

The dataset (`ml_ready_dataset.csv`) comes from the existing data-collection pipeline, which this project does not
rebuild.

---

## 1. Dataset requirements

The script needs a CSV with at least:

| Purpose | Config variable | Default column | Required |
|---|---|---|---|
| model input | `TITLE_COLUMN` | `title` | yes |
| target | `LABEL_COLUMN` | `label` (values `neutral` / `spike` / `crash`) | yes |
| time-based split | `TIMESTAMP_COLUMN` | `news_published_timestamp_utc` | recommended |
| reports only | `TICKER_COLUMN` | `ticker` | optional |

The script prints the file's real columns. If a configured column is missing, it tries common alternatives and tells
you which one it used. If nothing matches, it stops and lists the available columns so you know what to edit.

The current `ml_ready_dataset.csv` was inspected before writing this project:
- 390,057 rows and 26 columns; all four columns above exist.
- Labels: neutral 291,244 / spike 50,682 / crash 48,131.
- Timestamps range from 2024-09-24 to 2026-09-22, with none missing.
- Titles are 13 words on average.
- There are 157,131 repeated titles, because one article is stored once per ticker it mentions, and 20,329 titles
  carry conflicting labels.

**Title-level de-duplication (`INPUT_MODE = "title"`):**
- The model sees only the headline, so rows with the same headline are identical inputs. They are merged into one
  example.
- The label is decided by **majority vote**. Exact ties are dropped as ambiguous, and all counts are reported.
- This also guarantees that no headline appears in both the training and test sets.
- With `INPUT_MODE = "ticker_title"`, the model instead sees `"TICKER: headline"`, with one example per
  (headline, ticker).

## 2–6. Training in Google Colab

1. **Upload files to Google Drive**, e.g. to `MyDrive/finbert_market_effect/`:
   - `ml_ready_dataset.csv` (from `data/processed/`)
   - `train_finbert.py`
   - `requirements.txt`
2. **Enable a GPU:** *Runtime → Change runtime type → Hardware accelerator: GPU (T4)* → Save.
3. **Mount Drive and install dependencies**, in a new notebook cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !pip install -q -r /content/drive/MyDrive/finbert_market_effect/requirements.txt

   import torch
   print("GPU available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("GPU:", torch.cuda.get_device_name(0))
   ```
4. **Set the paths.** `DATA_PATH` and `OUTPUT_DIR` at the top of `train_finbert.py` default to the folder above.
   Change them if you used another folder, or pass them on the command line.
5. **Optional quick check (about 2 minutes).** This runs every stage on 3,000 rows so that problems show up before
   the long run:
   ```python
   !python /content/drive/MyDrive/finbert_market_effect/train_finbert.py --smoke-test --output-dir /content/smoke
   ```
6. **Train:**
   ```python
   !python /content/drive/MyDrive/finbert_market_effect/train_finbert.py
   ```
   The file is also divided into `# %%` cells, so you can paste it into notebook cells and run it step by step.

**Runtime:** after de-duplication there are roughly 210–230k unique headlines, about 155k of them for training. A
T4 takes very roughly 10–20 min per epoch at `MAX_LENGTH=64`, `batch=32` with fp16, so 4 epochs take about 1–1.5 h.
Early stopping may end sooner. These are estimates; the actual time is recorded in `metrics.json`.

**Colab disconnects:**
- Checkpoints are written to `OUTPUT_DIR/checkpoints/` on Drive, twice per epoch, keeping the last 2 plus the best.
- Rerun the same command and training resumes from the last checkpoint (`RESUME_FROM_CHECKPOINT = True`).
- Delete that folder to start fresh.

**Main settings**, all at the top of the script:

| Setting | Default | Why |
|---|---|---|
| `MAX_LENGTH` | 64 | Headlines are short; the script prints the token-length percentiles and the truncated share |
| `TRAIN_BATCH_SIZE` / `GRADIENT_ACCUMULATION_STEPS` | 32 / 1 | Fits a T4 with fp16. If you run out of memory, use 16 / 2 |
| `LEARNING_RATE`, `WARMUP_RATIO`, `WEIGHT_DECAY` | 2e-5, 6%, 0.01 | Standard BERT fine-tuning; AdamW with linear warm-up and decay |
| `NUM_EPOCHS`, `EARLY_STOPPING_PATIENCE` | 4, 3 checks | Validation runs twice per epoch; training stops when macro-F1 stops improving |
| `MAX_GRAD_NORM` | 1.0 | Gradient clipping |
| `CLASS_WEIGHTING` | balanced | See "Class imbalance" below |
| `REUSE_SENTIMENT_HEAD` | True | Starts from FinBERT's positive/negative/neutral output layer (positive→spike, negative→crash) |
| `SEED` | 42 | Seeds Python, NumPy, torch and the data order |

Also used:
- **Dynamic padding:** each batch is padded to its own longest title, which is faster than padding everything to
  `MAX_LENGTH`.
- **Attention masks** so padding is ignored.
- **fp16** mixed precision on GPU.
- **Best-model checkpointing** by validation macro-F1.
- FinBERT's built-in dropout (0.1).

**Class imbalance:** there are about 6 neutral examples for each spike or crash example. The script uses **one**
technique: **class-weighted cross-entropy**, with weights computed from the training split only.
- Weighted sampling would repeat the same minority headlines every epoch, which encourages memorising them, while
  wasting neutral examples.
- Focal loss is meant for much more extreme imbalance.
- The best checkpoint is chosen by **macro-F1**, so all three classes count equally.

## 7. Outputs

All paths are relative to `OUTPUT_DIR`:

```text
cleaned_dataset.csv            one row per unique input, with the resolved label
train.csv / validation.csv / test.csv
results/
    metrics.json               everything: data stats, split dates, model, time, GPU, best epoch, test metrics
    report.md                  human-readable summary, including the leakage and causality statements
    classification_report.csv  per-class precision / recall / F1 / support
    model_comparison.csv       TF-IDF + Logistic Regression vs fine-tuned FinBERT
    confusion_matrix.png/.csv  labelled Neutral / Spike / Crash (counts + row-normalised)
    confusion_summary.csv      which classes are confused most often
    test_predictions.csv       title, actual, predicted, confidence, prob_neutral, prob_spike, prob_crash, ...
    misclassified_examples.csv wrong predictions, most confident first (title, labels, confidence, ticker, timestamp)
    calibration.csv            confidence vs. actual accuracy per confidence bin
    training_history.csv/.png  training loss, validation loss / accuracy / macro-F1 per evaluation
    training_loss.png, validation_loss.png, accuracy.png
model/finbert_market_effect/   config.json, model.safetensors, tokenizer files, label_mapping.json
checkpoints/                   intermediate checkpoints (can be deleted after training)
```

## 8–9. Downloading the model and moving it to VS Code

1. In Google Drive, open `finbert_market_effect/model/`. Right-click `finbert_market_effect` and choose Download;
   Drive zips it, about 400 MB.

   Or zip it from Colab and download:
   ```python
   !cd /content/drive/MyDrive/finbert_market_effect/model && zip -r /content/finbert_market_effect.zip finbert_market_effect
   from google.colab import files; files.download('/content/finbert_market_effect.zip')
   ```
2. Unzip it into your local project so the layout is:
   ```text
   finbert_market_effect/
   ├── predict.py
   └── model/finbert_market_effect/   config.json, model.safetensors, tokenizer.json, tokenizer_config.json, label_mapping.json, ...
   ```

## 10–11. Running `predict.py` locally (CPU)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
python predict.py
```

```text
Enter news title: Apple reports record quarterly revenue
========================================
NEWS MARKET EFFECT PREDICTION
========================================

Title:
Apple reports record quarterly revenue

Prediction:
SPIKE

Confidence:
87.42%

Probabilities:
Neutral: 5.21%
Spike: 87.42%
Crash: 7.37%
========================================
```

The numbers above only illustrate the format. Real values depend on your trained model.

Other ways to use it:
- `python predict.py "headline one" "headline two"` predicts several headlines.
- `--json` prints raw JSON.
- `--model-dir PATH` uses a model stored elsewhere.

From Python, for example a Flask app:

```python
from predict import predict_news_effect
predict_news_effect("Apple reports record quarterly revenue")
# {'prediction': ..., 'confidence': ..., 'probabilities': {'neutral': ..., 'spike': ..., 'crash': ...}}
```

The model is loaded once and cached. Inference always runs on the CPU, and a headline takes roughly tens of
milliseconds.

## 12. Evaluation metrics

- **Accuracy:** share of correct predictions. It is misleading here, because always predicting "neutral" already
  scores about 75%. The report prints this majority-class figure next to the model's results.
- **Macro-F1:** average F1 of the three classes, each counted equally. This is the **main metric**.
- **Weighted-F1:** F1 averaged in proportion to each class's size.
- **Per class:** precision, recall, F1 and support, plus confusion matrices.
- **Baseline comparison:** TF-IDF + Logistic Regression is scored on the same test set. It is a benchmark only.
  `model_comparison.csv` shows the measured numbers; no model is declared better without them.

## 13. Important limitations

- **Headline only.** Most price moves have causes that are not in the headline, such as the wider market, other news,
  order flow or earnings details. Performance is inherently limited, and modest macro-F1 values are expected.
- **Label noise.** A headline that mentions several companies can be followed by different reactions for each. The
  majority vote simplifies this.
- **Time drift.** The test set is the most recent 15% of news. Market conditions change, and a model trained on
  2024–2026 news may degrade later.
- **Label definition.** The labels come from the data pipeline's rules: abnormal 5–60 minute returns versus the stock's
  own volatility, with volume confirmation. The model learns *that* definition, not "the truth".
- The model is not investment advice and should not be used for trading decisions.

## 14. Data-leakage prevention

- **The only model input is the title** (`TITLE → MARKET EFFECT`).
- The label was created by the data pipeline from **post-news** price and volume movement. That information exists
  only as the **target**. No future OHLCV, return, volume, volatility, `label_confidence` or other label-generation
  field is ever loaded as an input.
- **Chronological split:**
  - The oldest 70% of news is used for training, the next 15% for validation and the newest 15% for testing.
  - Nothing is shuffled before the split.
  - Rows with identical timestamps are never divided between sets.
- **Each unique headline appears in exactly one split.** The code asserts this.
- **Fitted on training data only:** the TF-IDF vocabulary and the class weights.
- **Chosen on validation:** hyperparameters, early stopping and the best checkpoint.
- **The test set is used exactly once**, for the final evaluation.

## 15. Model confidence vs actual probability

- `confidence` is the **softmax probability of the predicted class**: how strongly the model prefers that class over
  the other two.
- It is **not** a guaranteed probability that the stock will move. Class weighting, in particular, tends to push
  spike/crash probabilities upward.
- **How well the scores match reality** is measured on the test set:
  - `calibration.csv` compares mean confidence with actual accuracy in each confidence bin.
  - The expected calibration error (ECE) is recorded in `metrics.json`.
  - If the model is right about 60% of the time when it says 80%, its confidences are over-stated.
- The model predicts the dataset's **historical label**: whether stocks showed an unusual reaction after similar
  headlines. It does not show that a headline **caused** a price movement.
