"""Predict with the cf/ Bio_ClinicalBERT model + post-processing rule overrides.

After BERT predictions, we apply rule-based 1->0 overrides for sentences that
contain unambiguous normal-finding language. Examples:
  - "no acute cardiopulmonary process"
  - "within normal limits", "unremarkable"
  - "no new"/"no interval change"
  - "negative for"/"no evidence of" (only when no strong-pathology keywords)

Outputs predictions/test##-pred.csv with columns: row_id, prediction (and prob_1
for debugging).

Env vars:
  MODEL_DIR, THRESHOLD_PATH, TEST_GLOB, OUT_DIR, BATCH_SIZE,
  APPLY_OVERRIDES (1/0)
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/bert_model")
THRESHOLD_PATH = os.environ.get("THRESHOLD_PATH", "artifacts/threshold.json")
TEST_GLOB = os.environ.get("TEST_GLOB", "data/test*_text_only.csv")
OUT_DIR = os.environ.get("OUT_DIR", "predictions")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
APPLY_OVERRIDES = os.environ.get("APPLY_OVERRIDES", "1") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(THRESHOLD_PATH) as f:
    cfg = json.load(f)
THRESHOLD = float(cfg["threshold"])
MAX_LENGTH = int(cfg.get("max_length", 128))
TRUNC_SIDE = cfg.get("truncation_side", "right")
DROPOUT = float(cfg.get("dropout", 0.3))

print(f"DEVICE={DEVICE}  MODEL_DIR={MODEL_DIR}  THRESHOLD={THRESHOLD:.3f}  "
      f"MAX_LEN={MAX_LENGTH}  TRUNC={TRUNC_SIDE}  OVERRIDES={APPLY_OVERRIDES}")

# ----------------------------------------------------------------- model -----
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer.truncation_side = TRUNC_SIDE


class ClassifierHead(nn.Module):
    def __init__(self, hidden, n=2, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden, n)

    def forward(self, x):
        return self.linear(self.dropout(x))


class BertClassifier(nn.Module):
    def __init__(self, model_dir, dropout):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_dir, local_files_only=True)
        hidden = self.bert.config.hidden_size
        self.head = ClassifierHead(hidden, 2, dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.bert(**kwargs)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)


model = BertClassifier(MODEL_DIR, DROPOUT).to(DEVICE)
head_path = os.path.join(MODEL_DIR, "head.pt")
if os.path.exists(head_path):
    model.head.load_state_dict(torch.load(head_path, map_location=DEVICE))
    print(f"Loaded classifier head from {head_path}")
else:
    print(f"WARNING: {head_path} not found — head is randomly initialized.")
model.eval()


# --------------------------------------------------------- post-process ------
NORMAL_PATTERNS = [
    re.compile(r"\bwithin normal limits\b", re.I),
    re.compile(r"\bwnl\b", re.I),
    re.compile(r"\bunremarkable\b", re.I),
    re.compile(r"\bno new\b", re.I),
    re.compile(r"\bno interval change\b", re.I),
    re.compile(r"\bnormal sinus rhythm\b", re.I),
    re.compile(r"\b(?:lvef|ejection fraction).*?\b(?:5\d|6\d|7\d|>\s*5\d)\s*%?\b", re.I),
    re.compile(r"\bnormal systolic function\b", re.I),
    re.compile(r"\bno wall motion abnormalit\w*\b", re.I),
    re.compile(r"\bprobable degenerative\b", re.I),
    re.compile(r"\bpossible artifact\b", re.I),
]

# "no acute" override — narrow to specific words to avoid false matches.
NO_ACUTE = re.compile(
    r"\bno acute (cardiopulmonary|process|change|finding|distress)"
    r"(?:\s+(?:abnormalit\w*|disease))?", re.I,
)

# Strong-pathology keywords that DEFEAT a "no evidence of" override.
STRONG_PATHOLOGY = re.compile(
    r"\b(pneumonia|hemorrhage|fracture|hernias?|infarct|sepsis|"
    r"embolism|tumou?r|carcinoma|metastas|stroke|cva|"
    r"perforation|abscess|aneurysm|dissection|thrombosis|"
    r"obstruction|necrosis|fistula)\b", re.I,
)
NO_EVIDENCE = re.compile(r"\bno evidence of\b", re.I)


def should_override_to_zero(text):
    if not APPLY_OVERRIDES:
        return False
    t = str(text)
    for pat in NORMAL_PATTERNS:
        if pat.search(t):
            return True
    if NO_ACUTE.search(t):
        return True
    if NO_EVIDENCE.search(t) and not STRONG_PATHOLOGY.search(t):
        return True
    return False


# ----------------------------------------------------------------- predict ---
@torch.no_grad()
def predict_probs(texts):
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t if isinstance(t, str) and t.strip() else " "
                 for t in texts[i:i + BATCH_SIZE]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        kw = {k: enc[k] for k in enc if k in ("input_ids", "attention_mask", "token_type_ids")}
        logits = model(**kw)
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
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

    if APPLY_OVERRIDES:
        n_overridden = 0
        for i, t in enumerate(texts):
            if preds[i] == 1 and should_override_to_zero(t):
                preds[i] = 0
                n_overridden += 1
        print(f"  rule-based overrides applied: {n_overridden}")

    rid = df["row_id"].values if "row_id" in df.columns else np.arange(len(preds))
    out_df = pd.DataFrame({"row_id": rid, "prediction": preds, "prob_1": probs})
    name = os.path.basename(path).replace("_text_only.csv", "")
    out_path = os.path.join(OUT_DIR, f"{name}-pred.csv")
    out_df.to_csv(out_path, index=False)
    pos_rate = float(preds.mean()) if len(preds) else 0.0
    print(f"{name}: rows={len(preds)} pos_rate={pos_rate:.3f} -> {out_path}")

print("Prediction complete.")
