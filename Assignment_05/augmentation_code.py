import pandas as pd
import random
import nltk
import os

from nltk.corpus import wordnet

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# DOWNLOAD WORDNET TO C DRIVE
# ==========================================

nltk_data_path = os.path.join(
    os.environ["APPDATA"],
    "nltk_data"
)

os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path = [nltk_data_path]

nltk.download(
    "wordnet",
    download_dir=nltk_data_path
)

nltk.download(
    "omw-1.4",
    download_dir=nltk_data_path
)


# ==========================================
# 2. LOAD ORIGINAL DATASET
# ==========================================

df = pd.read_excel(
    "News_Classification_Labeling.xlsx"
)


# ==========================================
# 3. REMOVE MISSING VALUES
# ==========================================

df = df.dropna(
    subset=["Title", "Category"]
).reset_index(drop=True)


print("Total records:", len(df))


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Category"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


print("\nTraining records before augmentation:", len(train_df))
print("Testing records:", len(test_df))


# ==========================================
# 5. SYNONYM FUNCTION
# ==========================================

def get_synonym(word):

    synonyms = []

    for syn in wordnet.synsets(word):

        for lemma in syn.lemmas():

            synonym = lemma.name().replace("_", " ")

            if synonym.lower() != word.lower():

                synonyms.append(synonym)

    if synonyms:

        return random.choice(synonyms)

    return word


# ==========================================
# 6. AUGMENT TITLE
# ==========================================

def augment_title(title):

    words = title.split()

    valid_indices = []

    for i, word in enumerate(words):

        clean_word = word.strip(
            ".,!?;:()[]{}\"'"
        )

        if len(clean_word) > 3:

            valid_indices.append(i)

    if len(valid_indices) == 0:

        return title

    index = random.choice(valid_indices)

    original_word = words[index]

    clean_word = original_word.strip(
        ".,!?;:()[]{}\"'"
    )

    synonym = get_synonym(clean_word)

    if synonym.lower() != clean_word.lower():

        words[index] = original_word.replace(
            clean_word,
            synonym
        )

    return " ".join(words)


# ==========================================
# 7. SHOW ORIGINAL CATEGORY COUNTS
# ==========================================

print("\nCategory counts BEFORE augmentation:")

print(
    train_df["Category"].value_counts()
)


# ==========================================
# 8. DATA AUGMENTATION
# ==========================================

minority_classes = [
    "Health",
    "Energy"
]

target_count = 500

augmented_rows = []


for category in minority_classes:

    category_data = train_df[
        train_df["Category"] == category
    ]

    current_count = len(category_data)

    while current_count < target_count:

        random_row = category_data.sample(
            n=1,
            replace=True
        ).iloc[0]

        new_title = augment_title(
            random_row["Title"]
        )

        new_row = random_row.copy()

        new_row["Title"] = new_title

        augmented_rows.append(new_row)

        current_count += 1


# ==========================================
# 9. CREATE AUGMENTED TRAINING DATA
# ==========================================

augmented_df = pd.DataFrame(
    augmented_rows
)

final_train_df = pd.concat(
    [
        train_df,
        augmented_df
    ],
    ignore_index=True
)


# ==========================================
# 10. SHOW NEW CATEGORY COUNTS
# ==========================================

print("\nCategory counts AFTER augmentation:")

print(
    final_train_df["Category"].value_counts()
)


# ==========================================
# 11. CREATE TF-IDF FEATURES
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)


X_train = vectorizer.fit_transform(
    final_train_df["Title"]
)

X_test = vectorizer.transform(
    test_df["Title"]
)


y_train = final_train_df["Category"]

y_test = test_df["Category"]


print("\nTF-IDF features:", X_train.shape[1])


# ==========================================
# 12. CREATE LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(
    max_iter=2000
)


# ==========================================
# 13. TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# 14. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(
    X_test
)


# ==========================================
# 15. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)


# ==========================================
# 16. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()


print("\nClassification Report:")

print(report_df)


# ==========================================
# 17. CREATE PREDICTIONS TABLE
# ==========================================

predictions_df = pd.DataFrame({

    "Title":
        test_df["Title"].values,

    "Actual Category":
        y_test.values,

    "Predicted Category":
        y_pred
})


# ==========================================
# 18. CREATE SUMMARY
# ==========================================

summary_df = pd.DataFrame({

    "Item": [

        "Dataset",
        "Original Records",
        "Training Records Before Augmentation",
        "Training Records After Augmentation",
        "Testing Records",
        "TF-IDF Features",
        "Target Variable",
        "Number of Categories",
        "Train/Test Split",
        "Augmentation Method",
        "Augmented Categories",
        "Classification Algorithm",
        "Accuracy"
    ],

    "Result": [

        "News Classification",
        len(df),
        len(train_df),
        len(final_train_df),
        len(test_df),
        X_train.shape[1],
        "Category",
        y_train.nunique(),
        "80% / 20%",
        "Synonym Replacement",
        "Health and Energy",
        "Logistic Regression",
        f"{accuracy * 100:.2f}%"
    ]
})


# ==========================================
# 19. SAVE EVERYTHING TO EXCEL
# ==========================================

output_file = "Augmented_Logistic_Regression_Results.xlsx"


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


print("\n==========================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("Excel file created:")
print(output_file)
print("==========================================")