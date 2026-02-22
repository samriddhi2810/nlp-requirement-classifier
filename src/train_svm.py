import matplotlib
matplotlib.use("Agg")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("data/cleaned_requirements.csv")

X = df["clean_text"]
y = df["label"]

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 🔹 PIPELINE (TF-IDF + SVM together)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_df=0.9,
        min_df=2
    )),
    ("clf", LinearSVC(class_weight="balanced"))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Heapmap of Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["functional", "non-functional"])

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="rocket",
            xticklabels=["Functional", "Non-Functional"],
            yticklabels=["Functional", "Non-Functional"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (SVM + TF-IDF Bigrams)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# F1 bar chart
report = classification_report(y_test, y_pred, output_dict=True)

labels = ["Functional", "Non-Functional"]
f1_scores = [
    report["functional"]["f1-score"],
    report["non-functional"]["f1-score"]
]

plt.figure(figsize=(6,5))
plt.bar(labels, f1_scores,
        color=["#2E004F", "#FF4D6D"])
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("F1 Score per Class")
plt.tight_layout()
plt.savefig("f1_scores.png")
plt.close()