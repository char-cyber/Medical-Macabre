"""Predict ICD-codable labels for every data/test*_text_only.csv file using
the fine-tuned ClinicalBERT artifact produced by train.py.

Outputs predictions/test##-pred.csv with columns: row_id, prediction.

All knobs are env-var overridable:
  MODEL_DIR, THRESHOLD_PATH, TEST_GLOB, OUT_DIR, BATCH_SIZE
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/bert_model")
THRESHOLD_PATH = os.environ.get("THRESHOLD_PATH", "artifacts/threshold.json")
TEST_GLOB = os.environ.get("TEST_GLOB", "data/test*_text_only.csv")
OUT_DIR = os.environ.get("OUT_DIR", "predictions")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(THRESHOLD_PATH) as f:
    cfg = json.load(f)
THRESHOLD = float(cfg["threshold"])
MAX_LENGTH = int(cfg.get("max_length", 256))
TRUNC_SIDE = cfg.get("truncation_side", "left")

print(
    f"DEVICE={DEVICE}  MODEL_DIR={MODEL_DIR}  THRESHOLD={THRESHOLD:.3f}  "
    f"MAX_LEN={MAX_LENGTH}  TRUNC={TRUNC_SIDE}"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer.truncation_side = TRUNC_SIDE
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True
).to(DEVICE)
model.eval()


@torch.no_grad()
def predict_probs(texts):
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [
            t if isinstance(t, str) and t.strip() else " "
            for t in texts[i : i + BATCH_SIZE]
        ]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        logits = model(**enc).logits
        prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        out.append(prob)
    return np.concatenate(out) if out else np.array([])


os.makedirs(OUT_DIR, exist_ok=True)
files = sorted(glob.glob(TEST_GLOB))
if not files:
    raise SystemExit(f"No test files matched glob: {TEST_GLOB}")

for path in files:
    df = pd.read_csv(path)
    texts = df["text"].astype(str).fillna("").tolist()
    probs = predict_probs(texts)
    preds = (probs > THRESHOLD).astype(int)

    if "row_id" in df.columns:
        rid = df["row_id"].values
    else:
        rid = np.arange(len(preds))

    out_df = pd.DataFrame({"row_id": rid, "prediction": preds})
    name = os.path.basename(path).replace("_text_only.csv", "")
    out_path = os.path.join(OUT_DIR, f"{name}-pred.csv")
    out_df.to_csv(out_path, index=False)

    pos_rate = float(preds.mean()) if len(preds) else 0.0
    print(f"{name}: rows={len(preds)} pos_rate={pos_rate:.3f} -> {out_path}")

print("Prediction complete.")
