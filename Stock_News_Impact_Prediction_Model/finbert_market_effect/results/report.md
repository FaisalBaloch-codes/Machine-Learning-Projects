# FinBERT market-effect classifier - training report


- Generated: 2026-09-26T05:32:46+00:00  |  device: cuda  |  GPU: Tesla T4
- Model: `ProsusAI/finbert` (109,484,547 trainable parameters), head: reused FinBERT sentiment head (positive->spike, negative->crash, neutral->neutral)
- Training time: 29.6 min; best checkpoint at epoch 1.5 (validation macro-F1 0.5084)

## Data
- Source rows: 390,057; unique model inputs after cleaning/de-duplication: 221,828
- Classes (3): 0 = neutral, 1 = spike, 2 = crash
- Final class distribution: {'neutral': 162345, 'spike': 30379, 'crash': 29104}
- Headlines with conflicting labels across tickers/dates: 20,327 (resolved by majority vote; 10,988 exact ties dropped)
- Split: chronological (time-based)
  - train: 155,279 rows, ['2024-09-24 00:00', '2025-08-05 16:05'], {'neutral': 114850, 'spike': 20197, 'crash': 20232}
  - validation: 33,274 rows, ['2025-08-05 16:05', '2025-10-22 20:12'], {'neutral': 24577, 'spike': 4817, 'crash': 3880}
  - test: 33,275 rows, ['2025-10-22 20:12', '2026-09-22 16:25'], {'neutral': 22918, 'spike': 5365, 'crash': 4992}
- Class imbalance handling: class-weighted cross-entropy (balanced), weights {'neutral': 0.4507, 'spike': 2.5627, 'crash': 2.5583}

## Test results (untouched test set)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro precision | Macro recall |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 61.90% | 0.4953 | 0.6437 | 0.4893 | 0.5184 |
| Fine-tuned FinBERT | 66.40% | 0.5098 | 0.6695 | 0.5160 | 0.5292 |

Always predicting the majority class would score 68.87% accuracy, which is why macro-F1 (all classes weighted equally) is the main metric.
Expected calibration error of FinBERT confidences: 0.0677 (see calibration.csv).

## Data-leakage prevention
- **Model input = the news title only.** No price, return, volume, OHLCV, volatility, sentiment or label-
  generation field is given to the model. Post-news market movement was used by the data pipeline solely
  to create the target label (spike / crash / neutral).
- Chronological split: the model is trained on older news and tested on newer news, so it cannot learn from
  the future. Rows with the same timestamp are never split across sets.
- Each distinct headline appears in exactly one split (asserted in code).
- TF-IDF vocabulary and class weights are fitted on the training split only; hyper-parameters, early stopping
  and the best checkpoint are chosen on validation; the test set is used once, at the end.

## Interpretation
- The model predicts the dataset's *historical label definition*: whether the stock showed a statistically
  unusual reaction after similar headlines. It does **not** establish that a headline *causes* a price move.
- The confidence is the softmax probability of the predicted class. It is a model score, not a guaranteed
  probability that the market will move; check calibration.csv for how well it matches observed accuracy.