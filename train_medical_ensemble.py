import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

print("🚀 Script started")

# ============================
# CLEAN TEXT
# ============================
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ============================
# SPLIT NOTE → SENTENCES
# ============================
def split_note(text):
    parts = re.split(r"[.\n]", str(text))
    parts = [clean_text(p) for p in parts if len(p.strip()) > 5]
    return parts[:15] if parts else [clean_text(text)]

# ============================
# LOAD DATA
# ============================
def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    print(f"Loaded {len(df)} rows from {path}")
    return df

# ============================
# TRAIN CLINICAL BERT
# ============================
def train_bert(train_df, device):
    model_name = "emilyalsentzer/Bio_ClinicalBERT"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()

    batch_size = 4
    print("🧠 Training ClinicalBERT...")

    model.train()

    for epoch in range(1):
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = torch.tensor(labels[i:i+batch_size]).to(device)

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=batch_labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 200 == 0:
                print(f"Step {i} Loss: {loss.item():.4f}")

    return tokenizer, model

# ============================
# PREDICT PROBS
# ============================
def predict_probs(tokenizer, model, texts, device):
    model.eval()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]

    return probs.cpu().numpy()

# ============================
# NOTE PREDICTION (IMPROVED)
# ============================
def predict_notes(tokenizer, model, texts, threshold, device):
    preds = []

    for note in texts:
        sentences = split_note(note)

        if len(sentences) == 0:
            preds.append(0)
            continue

        probs = predict_probs(tokenizer, model, sentences, device)

        # 🔥 better aggregation (max instead of ratio)
        score = np.max(probs)

        preds.append(1 if score >= threshold else 0)

    return np.array(preds)

# ============================
# THRESHOLD TUNING
# ============================
def find_best_threshold(tokenizer, model, val_df, device):
    best_t = 0.5
    best_f1 = 0

    for t in np.arange(0.3, 0.8, 0.05):
        preds = predict_notes(tokenizer, model, val_df["text"], t, device)
        f1 = f1_score(val_df["label"], preds)

        print(f"Threshold {t:.2f} → F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print("✅ Best threshold:", best_t)
    return best_t

# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--predict-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("📂 Loading data...")
    df = load_data(args.train_file)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    print("🧠 Starting training...")
    tokenizer, model = train_bert(train_df, device)

    print("⚖️ Tuning threshold...")
    best_threshold = find_best_threshold(tokenizer, model, val_df, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔮 Generating predictions...")
    for file in args.predict_files:
        file_path = Path(file)

        print("Processing:", file_path)

        if not file_path.exists():
            print("❌ Missing:", file_path)
            continue

        test_df = pd.read_csv(file_path)

        preds = predict_notes(tokenizer, model, test_df["text"], best_threshold, device)

        out_path = output_dir / file_path.name.replace("_text_only", "_pred")

        pd.DataFrame({
            "row_id": test_df["row_id"],
            "label": preds
        }).to_csv(out_path, index=False)

        print("Saved:", out_path)

    print("🎉 DONE")

if __name__ == "__main__":
    main()