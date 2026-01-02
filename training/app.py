import tkinter as tk
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def classify_email():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        result_label.config(text="Please enter an email.")
        return

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    if prediction == "spam":
        result_label.config(text="⚠️ SPAM EMAIL", fg="red")
    else:
        result_label.config(text="✅ NOT SPAM", fg="green")

# UI window
root = tk.Tk()
root.title("Email Spam Detector")
root.geometry("400x300")

tk.Label(root, text="Paste Email Text:").pack()
input_text = tk.Text(root, height=8)
input_text.pack()

tk.Button(root, text="Check Email", command=classify_email).pack(pady=10)
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
