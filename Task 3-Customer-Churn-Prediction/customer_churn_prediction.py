# ==========================
# CUSTOMER CHURN PREDICTION
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --------------------------
# Load Dataset
# --------------------------

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# --------------------------
# Drop unnecessary columns
# --------------------------

df.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)

# --------------------------
# Encode categorical columns
# --------------------------

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Geography"] = le.fit_transform(df["Geography"])

# --------------------------
# Features and Target
# --------------------------

X = df.drop("Exited", axis=1)

y = df["Exited"]

# --------------------------
# Split Dataset
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------
# Logistic Regression
# --------------------------

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))

# --------------------------
# Random Forest
# --------------------------

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

# --------------------------
# Gradient Boosting
# --------------------------

gb = GradientBoostingClassifier(random_state=42)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

print("\nGradient Boosting Accuracy:")
print(accuracy_score(y_test, gb_pred))

# --------------------------
# Best Model
# --------------------------

models = {
    "Logistic Regression": accuracy_score(y_test, lr_pred),
    "Random Forest": accuracy_score(y_test, rf_pred),
    "Gradient Boosting": accuracy_score(y_test, gb_pred)
}

best = max(models, key=models.get)

print("\nBest Model:", best)

# --------------------------
# Classification Report
# --------------------------

print("\nClassification Report")

print(classification_report(y_test, rf_pred))

# --------------------------
# Confusion Matrix
# --------------------------

plt.figure(figsize=(6,5))

sns.heatmap(
    confusion_matrix(y_test, rf_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# --------------------------
# Feature Importance
# --------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.show()

# --------------------------
# Sample Prediction
# --------------------------

sample = X.iloc[[0]]

prediction = rf.predict(sample)

print("\nPrediction:")

if prediction[0] == 1:
    print("Customer will leave (Exited)")
else:
    print("Customer will stay")