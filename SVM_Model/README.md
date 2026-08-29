# SVM Classification Model (RBF Kernel) - News Category Classification

This project implements a Support Vector Machine (SVM) classifier with an **RBF (Radial Basis Function) kernel** for multi-class news article category classification.

---

## 📁 Project Structure

```
SVM_Model/
├── data/
│   ├── News_Classification_Labeling.csv     # Cleaned raw dataset (5,000 labeled articles)
│   ├── train_data.csv                       # 80% Stratified Training Split (4,000 samples)
│   ├── test_data.csv                        # 20% Stratified Testing Split (1,000 samples)
│   ├── train_tfidf.csv                      # TF-IDF Feature Matrix for Training set
│   ├── test_tfidf.csv                       # TF-IDF Feature Matrix for Testing set
│   ├── SVM_RBF_predictions.csv             # Model predictions aligned with Test Titles
│   ├── SVM_RBF_category_metrics.csv        # Per-category precision, recall, F1, accuracy
│   └── SVM_RBF_overall_summary.csv         # Overall performance metrics summary
├── tfidf calculation.py                     # Train/Test stratified split + TF-IDF vectorization
├── SVM model.py                             # SVM (RBF kernel) training and evaluation
└── README.md                                # Project documentation
```

---

## ⚙️ Methodology & Pipeline

1. **Train / Test Stratified Split**:
   - 80% Training set (4,000 samples) and 20% Testing set (1,000 samples).
   - Stratified sampling preserves category proportions across splits.
2. **TF-IDF Feature Extraction**:
   - Fitted **strictly** on the training titles to prevent data leakage (`min_df=2`, `max_df=0.95`).
   - Testing titles transformed using the training vectorizer vocabulary.
3. **Feature Scaling**:
   - `StandardScaler` fitted on training features and applied to test features.
4. **Model Training & Evaluation**:
   - `SVC(kernel="rbf", random_state=42)` trained on scaled features.
   - Evaluated on 1,000 unseen test samples across 6 news categories: `Business`, `Energy`, `Health`, `Markets`, `Politics`, `Technology`.

---

## 📊 Performance Summary

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **58.50%** |
| **Macro Precision** | **68.40%** |
| **Macro Recall** | **38.07%** |
| **Macro F1-Score** | **42.39%** |
| **Weighted F1-Score** | **56.54%** |

### Per-Category Performance

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Politics** | 87.76% | 49.71% | 63.47% | 173 |
| **Business** | 50.00% | 79.68% | 61.44% | 379 |
| **Markets** | 65.41% | 57.24% | 61.05% | 304 |
| **Technology** | 73.91% | 18.89% | 30.09% | 90 |
| **Energy** | 66.67% | 16.00% | 25.81% | 25 |
| **Health** | 66.67% | 6.90% | 12.50% | 29 |

---

## 🚀 How to Run

1. Navigate to the `SVM_Model` folder:
   ```bash
   cd SVM_Model
   ```

2. Run TF-IDF Feature Engineering:
   ```bash
   python "tfidf calculation.py"
   ```

3. Train and Evaluate the SVM Model:
   ```bash
   python "SVM model.py"
   ```
