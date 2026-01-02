import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# ===== Load dataset =====
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Remove empty rows
df = df.dropna()

print("✅ Dataset loaded")
print(df.head())

# ===== Split data =====
X = df["text"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== TF-IDF Vectorizer =====
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== Train model =====
model = MultinomialNB(alpha=0.3)
model.fit(X_train_vec, y_train)

# ===== Evaluate =====
pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)

print(f"🎯 Accuracy: {acc:.4f}")

# ===== Save model inside training folder =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "spam_model.pkl")
vec_path = os.path.join(BASE_DIR, "vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vec_path)

print("💾 Saved:", model_path)
print("💾 Saved:", vec_path)
