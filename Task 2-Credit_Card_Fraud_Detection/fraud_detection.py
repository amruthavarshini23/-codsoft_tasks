# ==========================================
# Credit Card Fraud Detection
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")

train = pd.read_csv("dataset/fraudTrain.csv")
test = pd.read_csv("dataset/fraudTest.csv")

print("Training Dataset Shape :", train.shape)
print("Testing Dataset Shape  :", test.shape)

# ==========================================
# Check Missing Values
# ==========================================

print("\nMissing Values:")
print(train.isnull().sum())

# ==========================================
# Drop Unnecessary Columns
# ==========================================

drop_columns = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "city",
    "state",
    "dob",
    "trans_num"
]

train.drop(columns=drop_columns, inplace=True, errors="ignore")
test.drop(columns=drop_columns, inplace=True, errors="ignore")

# ==========================================
# Encode Categorical Columns
# ==========================================

label_encoder = LabelEncoder()

categorical_columns = train.select_dtypes(include="object").columns

for column in categorical_columns:

    combined = pd.concat([train[column], test[column]], axis=0)

    label_encoder.fit(combined.astype(str))

    train[column] = label_encoder.transform(train[column].astype(str))
    test[column] = label_encoder.transform(test[column].astype(str))

# ==========================================
# Split Features and Target
# ==========================================

X_train = train.drop("is_fraud", axis=1)
y_train = train["is_fraud"]

X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]

# ==========================================
# Train Model
# ==========================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================================
# Prediction
# ==========================================

print("\nPredicting...")

y_pred = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("==============================")

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate","Fraud"],
    yticklabels=["Legitimate","Fraud"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# ==========================================
# Feature Importance
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))

importance.head(10).plot(kind="bar")

plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.show()

# ==========================================
# Sample Predictions
# ==========================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 20 Predictions\n")
print(results.head(20))

# ==========================================
# Save Predictions
# ==========================================

results.to_csv("prediction_output.csv", index=False)

print("\nPrediction file saved successfully!")
print("File Name : prediction_output.csv")

print("\nProject Completed Successfully!")