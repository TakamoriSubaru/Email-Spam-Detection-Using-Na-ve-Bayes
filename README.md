# Email Spam Detection Using Naïve Bayes

This project implements an **Email Spam Detection system** using **Artificial Intelligence (AI)** techniques. The system classifies email messages into **Spam** or **Non-Spam (Ham)** using the **Multinomial Naïve Bayes algorithm** combined with **TF-IDF feature extraction**.

This project was developed for an academic AI assignment and demonstrates the full machine learning pipeline: dataset preprocessing, feature extraction, model training, evaluation, and user interaction.

---

## 📌 Features
- Email text preprocessing
- TF-IDF feature extraction
- Multinomial Naïve Bayes classifier
- Accuracy evaluation
- Command-line email input for real-time classification
- Lightweight and fast execution

---

## 📊 Dataset
The dataset used in this project is obtained from **Kaggle** and consists of labeled email messages.

- **spam** → Unwanted or malicious emails  
- **ham** → Legitimate emails  

Dataset link:  
https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

---

## 🖼️ Screenshots

### Dataset Distribution
This figure shows the distribution of spam and non-spam (ham) emails in the dataset.

<img width="577" height="476" alt="Screenshot 2026-01-02 210410" src="https://github.com/user-attachments/assets/04056539-de9b-4550-bc31-76b995936bf3" />


### System Workflow
Overall workflow of the email spam detection system using TF-IDF and Naïve Bayes classification.

<img width="328" height="768" alt="Screenshot 2025-12-16 144423" src="https://github.com/user-attachments/assets/ed37fe15-d07b-4779-a6bb-a3ff74e4e5ed" />

### Email Classification Result
Example of email classification result using user input.

<img width="1919" height="273" alt="Screenshot 2026-01-02 210748" src="https://github.com/user-attachments/assets/1d621231-4627-4b11-af93-975ec947e9da" />

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```
git clone https://github.com/your-username/email-spam-detection.git
cd email-spam-detection

2️⃣ Create and activate virtual environment

python -m venv .venv

Windows (PowerShell):

.venv\Scripts\activate

If script execution is blocked:

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

3️⃣ Install dependencies

pip install -r requirements.txt

🚀 How to Run the Program
🔹 Train the Model

python training/train.py

Expected output:

Accuracy: 0.71
Saved: spam_model.pkl
Saved: vectorizer.pkl

🔹 Classify an Email

python training/classify_email.py

Example:

=== Email Spam Detector ===
Paste an email:
Congratulations! You won $1,000,000!
Result: Spam

🔹 Evaluate Model Performance

python training/evaluate.py
```
This script visualizes the model’s accuracy and evaluation metrics.
📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

🧠 Algorithm Used
- Multinomial Naïve Bayes
- TF-IDF (Term Frequency–Inverse Document Frequency)

🎓 Academic Purpose
This project is intended for:
- Artificial Intelligence coursework
- Machine learning demonstrations
- Educational research and experimentation

📎 References
- T. M. Mitchell, Machine Learning, McGraw-Hill, 1997
- A. McCallum and K. Nigam, “A comparison of event models for Naïve Bayes text classification,” 1998
- Kaggle Email Spam Classification Dataset

👤 Authors

Darryl Arief Tananjaya
Muhhamad Zharfan Agustiansyah
Selvi Lukman
