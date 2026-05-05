"""
train_clinicalbert.py
=====================
Fine-tunes Bio_ClinicalBERT AND trains a TF-IDF + Logistic Regression baseline
on the same train/val split.

Both models are saved:
  <output_dir>/best_model/          ← ClinicalBERT checkpoint (HuggingFace format)
  <output_dir>/lr_model/            ← Logistic Regression (sklearn pickle + vectorizer)

Usage (inside a SLURM job or interactively on a GPU node):
    python train_clinicalbert.py \
        --train_csv  prepared_data/train.csv \
        --val_csv    prepared_data/val.csv   \
        --output_dir /scratch/user/charu7465/baseline_results/clinicalbert \
        --epochs 4 \
        --batch_size 16 \
        --max_len 128 \
        --use_class_weights
"""

import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset (ClinicalBERT) ─────────────────────────────────────────────────────

class ClinicalSentenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts     = df['text'].tolist()
        self.labels    = df['label'].tolist()
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_class_weights(labels, num_classes=2):
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


def evaluate_bert(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            loss           = loss_fn(outputs.logits, labels)
            total_loss    += loss.item()
            preds          = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report   = classification_report(all_labels, all_preds,
                                     target_names=['not_codable', 'codable'],
                                     zero_division=0)
    return avg_loss, f1, report, all_preds


# ── Logistic Regression training ──────────────────────────────────────────────

def train_logistic_regression(train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               output_dir: str,
                               use_class_weights: bool = False):
    """
    Trains a TF-IDF (char+word ngrams) + Logistic Regression pipeline.
    Saves the fitted pipeline to <output_dir>/lr_model/pipeline.pkl.
    Returns val macro-F1.
    """
    print("\n" + "=" * 60)
    print("Training Logistic Regression baseline")
    print("=" * 60)

    lr_dir = os.path.join(output_dir, 'lr_model')
    os.makedirs(lr_dir, exist_ok=True)

    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    X_val   = val_df['text'].tolist()
    y_val   = val_df['label'].tolist()

    class_weight = 'balanced' if use_class_weights else None

    # Two vectorizers: word n-grams and char n-grams, concatenated via Pipeline
    # We use a single TfidfVectorizer with sublinear_tf for memory efficiency
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_features=100_000,
            strip_accents='unicode',
        )),
        ('lr', LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight=class_weight,
            solver='lbfgs',
            multi_class='auto',
            n_jobs=-1,
        )),
    ])

    print("  Fitting TF-IDF + LR pipeline...")
    pipeline.fit(X_train, y_train)

    val_preds = pipeline.predict(X_val)
    f1        = f1_score(y_val, val_preds, average='macro', zero_division=0)
    report    = classification_report(y_val, val_preds,
                                      target_names=['not_codable', 'codable'],
                                      zero_division=0)
    print(f"  Val Macro-F1: {f1:.4f}")
    print(report)

    # Save pipeline
    pkl_path = os.path.join(lr_dir, 'pipeline.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  LR pipeline saved to {pkl_path}")

    return f1


# ── ClinicalBERT training ──────────────────────────────────────────────────────

def train_clinicalbert(train_df, val_df, args, device):
    print("\n" + "=" * 60)
    print("Fine-tuning Bio_ClinicalBERT")
    print("=" * 60)

    hf_cache  = os.environ.get('TRANSFORMERS_CACHE', None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=hf_cache)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name, num_labels=2, cache_dir=hf_cache)
    model.to(device)

    train_dataset = ClinicalSentenceDataset(train_df, tokenizer, args.max_len)
    val_dataset   = ClinicalSentenceDataset(val_df,   tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    if args.use_class_weights:
        cw      = compute_class_weights(train_df['label'].tolist()).to(device)
        print(f"  Class weights: {cw.tolist()}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer    = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
                       optimizer, num_warmup_steps=warmup_steps,
                       num_training_steps=total_steps)

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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
            loss    = outputs.loss /args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} "
                      f"| Loss {running_loss/step:.4f}")

        val_loss, val_f1, report, _ = evaluate_bert(model, val_loader, device)
        print(f"\nEpoch {epoch} — Val loss: {val_loss:.4f} | Macro-F1: {val_f1:.4f}")
        print(report)

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  ✓ New best model saved to {best_path}  (F1={best_f1:.4f})\n")

    return best_f1, best_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv',   required=True)
    parser.add_argument('--val_csv',     required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--grad_accum',  default=2)
    parser.add_argument('--model_name',  default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--epochs',      type=int,   default=4)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--max_len',     type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=2e-5)
    parser.add_argument('--warmup_ratio',type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--skip_bert',   action='store_true',
                        help='Skip ClinicalBERT fine-tuning; train LR only.')
    parser.add_argument('--skip_lr',     action='store_true',
                        help='Skip Logistic Regression; fine-tune BERT only.')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    results = {}

    # ── Logistic Regression ──────────────────────────────────────────────────
    if not args.skip_lr:
        lr_f1 = train_logistic_regression(
            train_df, val_df, args.output_dir, args.use_class_weights)
        results['lr_val_macro_f1'] = lr_f1

    # ── ClinicalBERT ─────────────────────────────────────────────────────────
    if not args.skip_bert:
        bert_f1, best_path = train_clinicalbert(train_df, val_df, args, device)
        results['bert_val_macro_f1'] = bert_f1
        print(f"\nClinicalBERT best macro-F1: {bert_f1:.4f}")
        print(f"ClinicalBERT model saved at: {best_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("Training complete.")


if __name__ == '__main__':
    main()
