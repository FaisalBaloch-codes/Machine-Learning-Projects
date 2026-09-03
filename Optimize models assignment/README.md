# News Classification - Hyperparameter Optimization & Model Tuning

This repository contains the machine learning pipeline for **Hyperparameter Optimization** of Decision Tree and Random Forest classifiers applied to multi-class news article classification using TF-IDF feature extraction, 5-Fold Cross Validation, and GridSearchCV.

---

## 📌 Project Overview
- **Task:** Multi-class News Headline / Article Classification
- **Feature Extraction:** TF-IDF Vectorization (`min_df=2`, `max_df=0.95`, English stop-words removal)
- **Data Splitting:** 80% Training, 20% Testing (Stratified split with `random_state=42`)
- **Optimization Strategy:** Hyperparameter Tuning via **GridSearchCV** with **5-Fold Cross Validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`)
- **Models Optimized:**
  1. **Decision Tree Classifier**
  2. **Random Forest Classifier**

---

## 📂 Project Structure
```text
Optimize models assignment/
├── optimize_models.py                                     # End-to-end training & hyperparameter optimization script
├── News__Classification_Labeling.csv                      # Labeled news dataset
├── optimized_decisiontree_results.xlsx                    # Excel report for Decision Tree (Summary, Report, Predictions)
├── optimized_decisiontree_results_confusion_matrix.png    # Confusion matrix visualization for Decision Tree
├── optimized_randomforest_results.xlsx                    # Excel report for Random Forest (Summary, Report, Predictions)
├── optimized_randomforest_results_confusion_matrix.png    # Confusion matrix visualization for Random Forest
├── requirements.txt                                       # Python dependencies
└── README.md                                              # Project documentation
```

---

## ⚙️ Hyperparameter Search Grids

### 1. Decision Tree Classifier
```python
param_grid = {
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}
```

### 2. Random Forest Classifier
```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [20, None],
    "min_samples_split": [2, 5]
}
```

---

## 📊 Output Deliverables
Each model produces an Excel workbook containing three comprehensive sheets:
1. **Project Summary:** Metadata, dataset info, split configuration, best hyperparameters, 5-Fold CV accuracy, and final test accuracy.
2. **Classification Report:** Detailed Precision, Recall, F1-Score, and Support for each category.
3. **Predictions:** Full test set predictions comparing Actual vs Predicted categories alongside article titles.

Additionally, normalized confusion matrix plots are saved as `.png` files for visual evaluation.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Optimization Pipeline
```bash
python optimize_models.py
```
