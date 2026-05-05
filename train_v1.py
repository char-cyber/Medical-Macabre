import pandas as pd
import torch
import numpy as np
import joblib
import re

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "medicalai/ClinicalBERT"

print(f"Using device: {DEVICE}")

# -----------------------------
# SIMPLE NEGATION DETECTION
# -----------------------------
NEG_PAT = re.compile(
    r"\b(no|denies|without|negative for|rule out|not present|absent)\b",
    re.IGNORECASE
)

def apply_negation(label, text):
    if label == 1 and NEG_PAT.search(text):
        return 0
    return label

# -----------------------------
# LOAD DATA
# -----------------------------
train_df = pd.read_csv("data/train_data-text_and_labels.csv")
manual_df = pd.read_csv("data/manual_751.csv")

train_df = train_df.rename(columns={"text": "sentence"})
manual_df = manual_df.rename(columns={"text": "sentence"})

df = pd.concat([train_df, manual_df], ignore_index=True)

df["sentence"] = df["sentence"].astype(str)
df["label"] = df.apply(lambda x: apply_negation(x["label"], x["sentence"]), axis=1)

print(f"Training size: {len(df)}")

# -----------------------------
# SPLIT
# -----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["sentence"].tolist(),
    df["label"].tolist(),
    test_size=0.1,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# LOAD MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def embed(texts, batch_size=64):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(DEVICE)

            out = model(**inputs)
            vec = out.last_hidden_state.mean(dim=1)
            embs.append(vec.cpu())
    return torch.cat(embs).numpy()

# -----------------------------
# EMBEDDINGS
# -----------------------------
print("Embedding train...")
X_train = embed(train_texts)

print("Embedding val...")
X_val = embed(val_texts)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
weights = compute_class_weight("balanced", classes=[0,1], y=train_labels)
clf = LogisticRegression(max_iter=1000, class_weight={0:weights[0],1:weights[1]})

clf.fit(X_train, train_labels)

# -----------------------------
# THRESHOLD SEARCH
# -----------------------------
probs = clf.predict_proba(X_val)[:,1]

best_f1 = 0
best_t = 0.5

for t in np.arange(0.4, 0.7, 0.02):
    preds = (probs > t).astype(int)
    f1 = f1_score(val_labels, preds)

    print(f"t={t:.2f} → F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"\nBEST THRESHOLD: {best_t} | F1={best_f1}")

# -----------------------------
# SAVE
# -----------------------------
joblib.dump(clf, "model.pkl")
joblib.dump(best_t, "threshold.pkl")

model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

print("Training complete.")