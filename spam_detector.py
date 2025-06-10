import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data_path = "data/spam mail.csv"
if not os.path.exists(data_path):
    print("Data file not found.")
    exit()

df = pd.read_csv(data_path)
if 'Masseges' in df.columns:
    df.rename(columns={'Category': 'label', 'Masseges': 'message'}, inplace=True)
else:
    print("Expected column 'Masseges' not found. Check CSV file.")
    exit()

label_map = {'ham': 0, 'spam': 1}
df['label'] = df['label'].map(label_map)

X = df['message']
y = df['label']
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=1)

def clean_text(data_path):
    text = text.lower()  
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  
    text = re.sub(r'\d+', '', text)  
    text = text.strip()
    return text

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)
X_trn_vec = vectorizer.fit_transform(X_trn)
X_tst_vec = vectorizer.transform(X_tst)
nb = MultinomialNB()
nb.fit(X_trn_vec, y_trn)
nb_preds = nb.predict(X_tst_vec)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_trn_vec, y_trn)
lr_preds = lr.predict(X_tst_vec)

print("==Naive Bayes Results==")
print("Accuracy:", accuracy_score(y_tst, nb_preds))
print(classification_report(y_tst, nb_preds))

print("\n==Logistic Regression Results==")
print("Accuracy:", accuracy_score(y_tst, lr_preds))
print(classification_report(y_tst, lr_preds))
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_tst, nb_preds), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: NB")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_tst, lr_preds), annot=True, fmt="d", cmap="Purples")
plt.title("Confusion Matrix: LR")
plt.xlabel("Predicted")
plt.ylabel("Actual")

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
plt.tight_layout()
output_file = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(output_file)
print(f"Saved confusion matrices to {output_file}")
plt.show()
