# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Select required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels into numbers
# ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test custom messages
while True:
    sms = input("\nEnter an SMS message (or type 'exit' to quit): ")

    if sms.lower() == "exit":
        print("Program terminated.")
        break

    # Convert message into TF-IDF vector
    sms_tfidf = vectorizer.transform([sms])

    # Predict
    prediction = model.predict(sms_tfidf)

    # Display result
    if prediction[0] == 1:
        print("🚨 Spam Message")
    else:
        print("✅ Legitimate Message")