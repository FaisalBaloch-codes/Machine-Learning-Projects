import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD FILES
# ==========================================

tfidf_df = pd.read_csv("tfidf_matrix.csv")

original_df = pd.read_excel(
    "News_Classification_Labeling.xlsx"
)


# ==========================================
# 2. ADD CATEGORY
# ==========================================

df = tfidf_df.copy()

df["Category"] = original_df["Category"].values


# ==========================================
# 3. REMOVE MISSING CATEGORIES
# ==========================================

df = df.dropna(subset=["Category"])


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Category", "Title"])

y = df["Category"]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. CREATE DECISION TREE
# ==========================================

model = DecisionTreeClassifier(
    random_state=42
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification Report:")
print(report_df)


# ==========================================
# 11. PREDICTIONS TABLE
# ==========================================

test_indices = X_test.index

predictions_df = pd.DataFrame({
    "Title": df.loc[test_indices, "Title"].values,
    "Actual Category": y_test.values,
    "Predicted Category": y_pred
})


# ==========================================
# 12. PROJECT SUMMARY
# ==========================================

summary_df = pd.DataFrame({
    "Item": [
        "Dataset",
        "Total Records",
        "TF-IDF Features",
        "Target Variable",
        "Number of Categories",
        "Train/Test Split",
        "Training Records",
        "Testing Records",
        "Classification Algorithm",
        "Accuracy"
    ],

    "Result": [
        "News Classification",
        len(df),
        X.shape[1],
        "Category",
        y.nunique(),
        "80% / 20%",
        len(X_train),
        len(X_test),
        "Decision Tree Classifier",
        f"{accuracy * 100:.2f}%"
    ]
})


# ==========================================
# 13. SAVE RESULTS TO EXCEL
# ==========================================

output_file = "Decision_Tree_Results.xlsx"

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Project Summary",
        index=False
    )

    report_df.to_excel(
        writer,
        sheet_name="Classification Report"
    )

    predictions_df.to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )


print("\n================================")
print("Decision Tree completed!")
print("Excel file:", output_file)
print("================================")