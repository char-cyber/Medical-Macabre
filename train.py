"""Bio_ClinicalBERT fine-tune with stratified 5-fold CV + multi-seed search,
then retrain on the full set with the best seed for 30 epochs.

Architecture:
  Bio_ClinicalBERT base -> Dropout(0.3) -> Linear(768 -> 2)

Training:
  - Differential learning rates: BERT layers 2e-5, classifier head 1e-3
  - Linear warmup (10%) + AdamW (weight_decay=0.01)
  - Gradient clipping 1.0
  - Max sequence length: 128 tokens (truncate from the right by default,
    flip to 'left' via TRUNC_SIDE=left if you want tail-keeping)
  - CUDA determinism enforced for reproducibility

Workflow:
  1) Stratified 5-fold CV across seeds {42,43,44,45}.
  2) Pick the seed with the best mean fold F1 on the held-out fold.
  3) Retrain on the FULL combined_train.csv with the best seed for FINAL_EPOCHS.

Inputs:
  data/combined/combined_train.csv  (text,label)

Outputs:
  artifacts/bert_model/             (model + tokenizer)
  artifacts/threshold.json          (single-threshold predict metadata)
  artifacts/cv_summary.json         (per-seed/fold F1)

Env vars:
  MODEL_DIR, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, FOLDS,
  CV_EPOCHS, FINAL_EPOCHS, SEEDS, LR_BERT, LR_HEAD, DROPOUT,
  THRESHOLD_DEFAULT, TRUNC_SIDE
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

# ---------------------------------------------------------------- config -----
MODEL_NAME = os.environ.get("MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT")
MODEL_DIR = os.environ.get("MODEL_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "artifacts/bert_model")
THRESHOLD_PATH = os.environ.get("THRESHOLD_PATH", "artifacts/threshold.json")
CV_PATH = os.environ.get("CV_PATH", "artifacts/cv_summary.json")
TRAIN_CSV = os.environ.get("TRAIN_CSV", "data/combined/combined_train.csv")

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
FOLDS = int(os.environ.get("FOLDS", "5"))
CV_EPOCHS = int(os.environ.get("CV_EPOCHS", "8"))
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", "30"))
LR_BERT = float(os.environ.get("LR_BERT", "2e-5"))
LR_HEAD = float(os.environ.get("LR_HEAD", "1e-3"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.10"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,43,44,45").split(",")]
TRUNC_SIDE = os.environ.get("TRUNC_SIDE", "right")
THRESHOLD_DEFAULT = float(os.environ.get("THRESHOLD_DEFAULT", "0.50"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------- load data ---
print(f"DEVICE={DEVICE}  MODEL={MODEL_DIR or MODEL_NAME}  MAX_LEN={MAX_LENGTH}  "
      f"BS={BATCH_SIZE}  FOLDS={FOLDS}  CV_EPOCHS={CV_EPOCHS}  FINAL_EPOCHS={FINAL_EPOCHS}")

df = pd.read_csv(TRAIN_CSV)
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]
df["label"] = df["label"].astype(int).clip(0, 1)
print(f"Total training rows: {len(df)}  | label counts: {df['label'].value_counts().to_dict()}")

# ------------------------------------------------------------- tokenizer/m ---
tokenizer_src = MODEL_DIR or MODEL_NAME
is_local = bool(MODEL_DIR) and os.path.isdir(MODEL_DIR)
print(f"Tokenizer/encoder source: {tokenizer_src} (local_files_only={is_local})")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=is_local)
tokenizer.truncation_side = TRUNC_SIDE


class ClassifierHead(nn.Module):
    def __init__(self, hidden, n_classes=2, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.linear(self.dropout(x))


class BertClassifier(nn.Module):
    def __init__(self, bert_src, dropout=0.3, local=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_src, local_files_only=local)
        hidden = self.bert.config.hidden_size
        self.head = ClassifierHead(hidden, 2, dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.bert(**kwargs)
        # Use [CLS] pooled output if available, else mean over tokens.
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)


# ------------------------------------------------------------- dataset -------
class SentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], int(self.labels[i])


def collate(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    enc = tokenizer(
        texts,
        padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def make_optimizer(model, lr_bert, lr_head, wd):
    bert_params = [p for n, p in model.named_parameters() if n.startswith("bert.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    return AdamW(
        [
            {"params": bert_params, "lr": lr_bert, "weight_decay": wd},
            {"params": head_params, "lr": lr_head, "weight_decay": wd},
        ]
    )


def train_one_run(train_idx, val_idx, seed, epochs, log_prefix=""):
    set_seed(seed)
    train_ds = SentDataset(df["text"].iloc[train_idx], df["label"].iloc[train_idx])
    val_ds = SentDataset(df["text"].iloc[val_idx], df["label"].iloc[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = BertClassifier(tokenizer_src, dropout=DROPOUT, local=is_local).to(DEVICE)

    counts = np.bincount(df["label"].iloc[train_idx].values, minlength=2)
    total = counts.sum()
    weights = torch.tensor(
        [total / (2.0 * max(counts[0], 1)), total / (2.0 * max(counts[1], 1))],
        dtype=torch.float,
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    total_steps = max(1, len(train_loader) * epochs)
    opt = make_optimizer(model, LR_BERT, LR_HEAD, WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * WARMUP_RATIO), total_steps)

    best_f1 = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**{k: batch[k] for k in batch if k in ("input_ids", "attention_mask", "token_type_ids")})
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()

        # eval
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels")
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**{k: batch[k] for k in batch if k in ("input_ids", "attention_mask", "token_type_ids")})
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_p.append(p); all_l.append(labels.numpy())
        probs = np.concatenate(all_p) if all_p else np.array([])
        labels = np.concatenate(all_l) if all_l else np.array([])
        preds = (probs > THRESHOLD_DEFAULT).astype(int)
        if len(labels):
            f1 = f1_score(labels, preds, zero_division=0)
            acc = accuracy_score(labels, preds)
        else:
            f1 = acc = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{log_prefix} epoch {ep+1}/{epochs}  F1={f1:.4f}  Acc={acc:.4f}")

    return best_f1, best_state


# ---------------------------------------------------------- 5-fold CV --------
print("\n=== Stratified 5-fold CV across seeds:", SEEDS, "===")
cv_results = {}  # {seed: {"folds": [f1...], "mean": ...}}
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    fold_f1s = []
    for fi, (tr, va) in enumerate(skf.split(df["text"], df["label"])):
        f1, _ = train_one_run(tr, va, seed=seed, epochs=CV_EPOCHS,
                              log_prefix=f"[seed {seed} fold {fi+1}]")
        fold_f1s.append(f1)
    mean = float(np.mean(fold_f1s))
    cv_results[seed] = {"folds": [float(x) for x in fold_f1s], "mean": mean}
    print(f"seed={seed}: mean CV F1={mean:.4f}  folds={fold_f1s}")

best_seed = max(cv_results, key=lambda s: cv_results[s]["mean"])
print(f"\nBEST SEED: {best_seed}  mean CV F1={cv_results[best_seed]['mean']:.4f}")

# ---------------------------------------------------------- final retrain ----
print(f"\n=== Final retrain on ALL {len(df)} rows for {FINAL_EPOCHS} epochs (seed={best_seed}) ===")

# We split a tiny holdout (5%) for threshold-tuning + best-checkpoint selection.
all_idx = np.arange(len(df))
tr_idx, ho_idx = train_test_split(all_idx, test_size=0.05, random_state=best_seed,
                                  stratify=df["label"])
best_f1, best_state = train_one_run(
    tr_idx, ho_idx, seed=best_seed, epochs=FINAL_EPOCHS,
    log_prefix=f"[FINAL seed={best_seed}]"
)

# Reload best state, save model.
final_model = BertClassifier(tokenizer_src, dropout=DROPOUT, local=is_local).to(DEVICE)
final_model.load_state_dict(best_state)
final_model.eval()

# Find a good threshold by sweeping on the holdout
ho_ds = SentDataset(df["text"].iloc[ho_idx], df["label"].iloc[ho_idx])
ho_loader = DataLoader(ho_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
all_p, all_l = [], []
with torch.no_grad():
    for batch in ho_loader:
        labels = batch.pop("labels")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = final_model(**{k: batch[k] for k in batch if k in ("input_ids", "attention_mask", "token_type_ids")})
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_p.append(p); all_l.append(labels.numpy())
probs = np.concatenate(all_p) if all_p else np.array([])
labels = np.concatenate(all_l) if all_l else np.array([])
best_t, best_t_f1 = 0.5, -1.0
for t in np.arange(0.20, 0.80, 0.02):
    preds = (probs > t).astype(int)
    if preds.sum() == 0 or preds.sum() == len(preds):
        continue
    f1 = f1_score(labels, preds, zero_division=0)
    if f1 > best_t_f1:
        best_t_f1, best_t = f1, float(t)

print(f"\nFinal holdout best threshold={best_t:.3f}  F1={best_t_f1:.4f}")

# ---------------------------------------------------------------- save -------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(THRESHOLD_PATH) or ".", exist_ok=True)

# Save Bert + tokenizer separately so predict.py can load it via AutoTokenizer/AutoModel
final_model.bert.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save the head weights into the same directory.
torch.save(final_model.head.state_dict(), os.path.join(OUTPUT_DIR, "head.pt"))

with open(THRESHOLD_PATH, "w") as f:
    json.dump({
        "threshold": best_t,
        "max_length": MAX_LENGTH,
        "truncation_side": TRUNC_SIDE,
        "model_dir": OUTPUT_DIR,
        "best_seed": best_seed,
        "cv_mean_f1": cv_results[best_seed]["mean"],
        "holdout_f1": best_t_f1,
        "dropout": DROPOUT,
    }, f, indent=2)

with open(CV_PATH, "w") as f:
    json.dump({"seeds": cv_results, "best_seed": best_seed,
               "best_threshold": best_t, "holdout_f1": best_t_f1}, f, indent=2)

print(f"\nSaved encoder+tokenizer to {OUTPUT_DIR}")
print(f"Saved head.pt to {OUTPUT_DIR}/head.pt")
print(f"Saved threshold metadata to {THRESHOLD_PATH}")
print(f"Saved CV summary to {CV_PATH}")
print("Training complete.")
