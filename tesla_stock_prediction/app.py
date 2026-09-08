# ============================================================
# TESLA STOCK PRICE PREDICTION USING RNN
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense


# ============================================================
# 1. SETTINGS
# ============================================================

FILE_PATH = "Tesla.csv"

LOOKBACK = 60

RESULTS_FOLDER = Path("results")
RESULTS_FOLDER.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(FILE_PATH)

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CONVERT DATE COLUMN
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

# Sort according to date
df = df.sort_values("Date")

# Reset index
df = df.reset_index(drop=True)


# ============================================================
# 4. SELECT CLOSE PRICE
# ============================================================

data = df[["Date", "Close"]].copy()

# Remove missing values
data = data.dropna(subset=["Close"])

print("\nSelected data:")
print(data.head())

print("\nNumber of rows:", len(data))


# ============================================================
# 5. EXTRACT CLOSE PRICES
# ============================================================

prices = data["Close"].values.reshape(-1, 1)


# ============================================================
# 6. CREATE X AND Y SEQUENCES
# ============================================================

X = []
Y = []

for i in range(LOOKBACK, len(prices)):

    # Previous 60 days
    X.append(prices[i - LOOKBACK:i])

    # Next day's closing price
    Y.append(prices[i])


X = np.array(X)
Y = np.array(Y)


print("\n========================================")
print("X AND Y")
print("========================================")

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ============================================================
# 7. SPLIT DATA INTO 80% TRAINING AND 20% TEMPORARY
# ============================================================

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.20, shuffle=False)


# ============================================================
# 8. SPLIT TEMPORARY DATA INTO 10% VALIDATION
#    AND 10% TESTING
# ============================================================

X_val, X_test, Y_val, Y_test = train_test_split(X_temp,Y_temp, test_size=0.50,shuffle=False)


print("\n========================================")
print("DATA SPLIT")
print("========================================")

print("Training data:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("\nValidation data:")
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ============================================================
# 9. SCALE THE DATA
# ============================================================

# Create scaler
scaler = MinMaxScaler(feature_range=(0, 1))


# Fit scaler ONLY on training data
X_train_2d = X_train.reshape(-1, 1)

scaler.fit(X_train_2d)


# Function to scale X while keeping 3D shape
def scale_X(X_data):
    original_shape = X_data.shape

    X_2d = X_data.reshape(-1, 1)

    X_scaled = scaler.transform(X_2d)

    return X_scaled.reshape(original_shape)


# Scale X
X_train = scale_X(X_train)
X_val = scale_X(X_val)
X_test = scale_X(X_test)


# Scale Y
Y_train = scaler.transform(
    Y_train.reshape(-1, 1)
)

Y_val = scaler.transform(
    Y_val.reshape(-1, 1)
)

Y_test = scaler.transform(
    Y_test.reshape(-1, 1)
)


print("\n========================================")
print("AFTER SCALING")
print("========================================")

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# ============================================================
# 10. BUILD RNN MODEL
# ============================================================

model = Sequential([

    # Input
    Input(shape=(LOOKBACK, 1)),

    # -----------------------------
    # RNN Layer 1
    # -----------------------------
    SimpleRNN(
        64,
        return_sequences=True
    ),

    # -----------------------------
    # RNN Layer 2
    # -----------------------------
    SimpleRNN(
        32,
        return_sequences=False
    ),

    # -----------------------------
    # Dense Layer 1
    # -----------------------------
    Dense(
        64,
        activation="relu"
    ),

    # -----------------------------
    # Dense Layer 2
    # -----------------------------
    Dense(
        32,
        activation="relu"
    ),

    # -----------------------------
    # Dense Layer 3
    # -----------------------------
    Dense(
        16,
        activation="relu"
    ),

    # -----------------------------
    # Output Layer
    # 1 cell
    # -----------------------------
    Dense(1)
])


# ============================================================
# 11. DISPLAY MODEL
# ============================================================

print("\n========================================")
print("MODEL SUMMARY")
print("========================================")

model.summary()


# ============================================================
# 12. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n========================================")
print("TRAINING MODEL")
print("========================================")

history = model.fit(

    X_train,
    Y_train,

    epochs=50,

    batch_size=32,

    # Explicit validation dataset
    validation_data=(X_val, Y_val),

    verbose=1
)


# ============================================================
# 14. PREDICT TEST DATA
# ============================================================

print("\n========================================")
print("PREDICTING TEST DATA")
print("========================================")

predictions_scaled = model.predict(X_test)


# ============================================================
# 15. CONVERT PREDICTIONS BACK TO REAL PRICES
# ============================================================

Y_test_original = scaler.inverse_transform(
    Y_test
)

predictions_original = scaler.inverse_transform(
    predictions_scaled
)


# ============================================================
# 16. CALCULATE EVALUATION METRICS
# ============================================================

mae = mean_absolute_error(
    Y_test_original,
    predictions_original
)

rmse = np.sqrt(
    mean_squared_error(
        Y_test_original,
        predictions_original
    )
)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"MAE  : {mae:.2f}")

print(f"RMSE : {rmse:.2f}")


# ============================================================
# 17. GET TEST DATES
# ============================================================

# Find dates corresponding to Y_test
test_start_index = len(X_train) + len(X_val)

test_dates = data["Date"].iloc[
    LOOKBACK + test_start_index:
].reset_index(drop=True)


# ============================================================
# 18. CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame({

    "Date": test_dates,

    "Actual_Close":
        Y_test_original.flatten(),

    "Predicted_Close":
        predictions_original.flatten()
})


print("\n========================================")
print("PREDICTION RESULTS")
print("========================================")

print(results.head(10))


# ============================================================
# 19. SAVE PREDICTIONS
# ============================================================

prediction_file = (
    RESULTS_FOLDER / "tesla_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False
)

print("\nPredictions saved to:")

print(prediction_file)


# ============================================================
# 20. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Tesla RNN Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()


loss_plot = (
    RESULTS_FOLDER / "training_validation_loss.png"
)

plt.savefig(
    loss_plot,
    dpi=300
)

plt.show()


# ============================================================
# 21. PLOT ACTUAL VS PREDICTED PRICES
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual_Close"],
    label="Actual Close"
)

plt.plot(
    results["Date"],
    results["Predicted_Close"],
    label="Predicted Close"
)

plt.title(
    "Tesla Stock Price Prediction Using RNN"
)

plt.xlabel("Date")

plt.ylabel("Closing Price")

plt.legend()

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()


prediction_plot = (
    RESULTS_FOLDER / "actual_vs_predicted.png"
)

plt.savefig(
    prediction_plot,
    dpi=300
)

plt.show()


# ============================================================
# 22. SAVE MODEL
# ============================================================

model_file = (
    RESULTS_FOLDER / "tesla_rnn_model.keras"
)

model.save(model_file)

print("\nModel saved to:")

print(model_file)


# ============================================================
# 23. FINAL INFORMATION
# ============================================================

print("\n========================================")
print("ASSIGNMENT COMPLETED")
print("========================================")

print("\nData split:")
print("80% Training")
print("10% Validation")
print("10% Testing")

print("\nNetwork:")
print("2 RNN layers")
print("3 Dense hidden layers")
print("1 Output neuron")

print("\nResults folder:")
print(RESULTS_FOLDER)