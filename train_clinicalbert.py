"""
train_clinicalbert.py
Fine-tunes Bio_ClinicalBERT on your ICD-codability classification task.
Designed for TAMU Grace with the exact library versions installed in the setup.

Usage (inside a SLURM job or interactively on a GPU node):
    python train_clinicalbert.py \
        --train_csv  prepared_data/train.csv \
        --val_csv    prepared_data/val.csv   \
        --output_dir /scratch/user/charu7465/baseline_results/clinicalbert \
        --epochs 4 \
        --batch_size 16 \
        --max_len 128
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW                          # torch's own AdamW (stable)

# transformers 4.40.0 — no get_linear_schedule_with_warmup import change needed
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ClinicalSentenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training helpers ──────────────────────────────────────────────────────────

def compute_class_weights(labels, num_classes=2):
    """Inverse-frequency class weights as a float tensor."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes      # normalise
    return torch.tensor(weights, dtype=torch.float)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds,
                                   target_names=['not_codable', 'codable'],
                                   zero_division=0)
    return avg_loss, f1, report, all_preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv',   required=True)
    parser.add_argument('--val_csv',     required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--model_name',  default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--epochs',      type=int,   default=4)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--max_len',     type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=2e-5)
    parser.add_argument('--warmup_ratio',type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use inverse-frequency loss weights for imbalanced data.')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ────────────────────────────────────────────────────────────
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    # ── Tokenizer & model ────────────────────────────────────────────────────
    hf_cache = os.environ.get('TRANSFORMERS_CACHE', None)
    print(f"HF cache: {hf_cache}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=hf_cache,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        cache_dir=hf_cache,
    )
    model.to(device)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_dataset = ClinicalSentenceDataset(train_df, tokenizer, args.max_len)
    val_dataset   = ClinicalSentenceDataset(val_df,   tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Loss with optional class weights ─────────────────────────────────────
    if args.use_class_weights:
        cw = compute_class_weights(train_df['label'].tolist()).to(device)
        print(f"Class weights: {cw.tolist()}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1   = 0.0
    best_path = os.path.join(args.output_dir, 'best_model')

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} "
                      f"| Loss {running_loss/step:.4f}")

        # ── Validation ───────────────────────────────────────────────────────
        val_loss, val_f1, report, _ = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch} — Val loss: {val_loss:.4f}  |  Macro-F1: {val_f1:.4f}")
        print(report)

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  ✓ New best model saved to {best_path}  (F1={best_f1:.4f})\n")

    print(f"\nTraining complete. Best macro-F1 on validation: {best_f1:.4f}")
    print(f"Best model saved at: {best_path}")


if __name__ == '__main__':
    main()
