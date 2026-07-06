# ==========================================
# MOVIE GENRE CLASSIFICATION
# IMDb Dataset (train_data.txt)
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# Step 1 : Load Dataset
# ==========================================

print("Loading Dataset...")

df = pd.read_csv(
    "Genre Classification Dataset/train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# ==========================================
# Step 2 : Features and Target
# ==========================================

X = df["DESCRIPTION"]
y = df["GENRE"]

# ==========================================
# Step 3 : Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Step 4 : TF-IDF
# ==========================================

print("\nConverting Text into TF-IDF...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# ==========================================
# Step 5 : Train Model
# ==========================================

print("\nTraining Model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==========================================
# Step 6 : Prediction
# ==========================================

print("\nPredicting...")

y_pred = model.predict(X_test)

# ==========================================
# Step 7 : Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("===================================")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================
# Step 8 : Predict Your Own Movie
# ==========================================

while True:

    movie = input("\nEnter Movie Description (type exit to stop): ")

    if movie.lower() == "exit":
        break

    movie_vector = tfidf.transform([movie])

    prediction = model.predict(movie_vector)

    print("\nPredicted Genre :", prediction[0])

print("\nProgram Finished.")