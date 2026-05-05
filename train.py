import os, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

MODEL_NAME = "bert_model"  # local model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 256

# ---------------- DATA ----------------
df1 = pd.read_csv("data/train_data-text_and_labels.csv")
df2 = pd.read_csv("data/new_train.csv")

df = pd.concat([df1, df2]).drop_duplicates(subset="text")

texts = df["text"].astype(str).fillna("").tolist()
labels = df["label"].astype(int).tolist()

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
tokenizer.truncation_side = "left"

# ---------------- DATASET ----------------
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=BATCH_SIZE)

# ---------------- MODEL ----------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    local_files_only=True
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------- TRAIN ----------------
best_f1 = 0
best_thresh = 0.5

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # -------- VALIDATION --------
    model.eval()
    probs, true = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            true.extend(batch["labels"].cpu().numpy())

    probs = np.array(probs)
    true = np.array(true)

    # 🔥 threshold sweep
    for t in np.linspace(0.2, 0.5, 20):
        preds = (probs > t).astype(int)
        f1 = f1_score(true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    acc = accuracy_score(true, (probs > best_thresh).astype(int))

    print(f"Epoch {epoch+1} | Acc={acc:.4f} F1={best_f1:.4f} Thresh={best_thresh:.3f}")

# ---------------- SAVE ----------------
os.makedirs("artifacts/bert_model", exist_ok=True)
model.save_pretrained("artifacts/bert_model")
tokenizer.save_pretrained("artifacts/bert_model")

with open("artifacts/threshold.json", "w") as f:
    json.dump({
        "threshold": float(best_thresh),
        "max_length": MAX_LEN,
        "truncation_side": "left"
    }, f, indent=2)

print("Training complete.")