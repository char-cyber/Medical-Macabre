import pandas as pd
import torch
import glob
import os
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("bert_model", local_files_only=True)
model = AutoModel.from_pretrained("bert_model", local_files_only=True).to(DEVICE)
model.eval()

clf = joblib.load("model.pkl")
threshold = joblib.load("threshold.pkl")

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

os.makedirs("predictions", exist_ok=True)

for file in sorted(glob.glob("data/test*_text_only.csv")):
    print(f"Processing {file}")

    df = pd.read_csv(file)
    texts = df["text"].astype(str).tolist()

    X = embed(texts)
    probs = clf.predict_proba(X)[:,1]

    preds = (probs > threshold).astype(int)

    test_id = os.path.basename(file).replace("_text_only.csv", "")

    out = pd.DataFrame({
        "row_id": range(len(preds)),
        "prediction": preds
    })

    out.to_csv(f"predictions/{test_id}-pred.csv", index=False)
    print(f"Saved {test_id}")