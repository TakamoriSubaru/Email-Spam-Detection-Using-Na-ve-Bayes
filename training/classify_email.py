import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "spam_model.pkl")
vec_path = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load model + vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vec_path)

print("\n=== Email Spam Detector ===\n")

while True:
    text = input("Paste an email (or type 'exit'): ")

    if text.lower() == "exit":
        print("👋 Goodbye!")
        break

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    label = "🚨 SPAM" if pred == "spam" else "✅ Not Spam"

    print(f"{label}  (Confidence: {prob:.2f})\n")


#.venv\Scripts\activate
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#cd training
#python train.py
#python classify_email.py
#Congratulations! You've won $5000. Click here to claim now!

