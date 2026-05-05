"""
predict.py
==========
Runs inference with EITHER (or both):
  - The saved ClinicalBERT checkpoint  (<model_dir>/best_model/)
  - The saved Logistic Regression pipeline (<model_dir>/lr_model/pipeline.pkl)

Writes  testXX-pred.csv  files (columns: id, label) for the Gradescope autograder.
If both models are available and --ensemble is set, predictions are combined by
majority vote (LR + BERT logits vote; BERT wins ties).

Usage:
    # BERT only
    python predict.py \
        --model_dir /scratch/user/charu7465/baseline_results/clinicalbert \
        --test_files test1_text_only.csv test2_text_only.csv \
        --output_dir ./predictions

    # LR only
    python predict.py \
        --model_dir /scratch/user/charu7465/baseline_results/clinicalbert \
        --test_files test1_text_only.csv \
        --output_dir ./predictions \
        --use_lr_only

    # Ensemble
    python predict.py \
        --model_dir /scratch/user/charu7465/baseline_results/clinicalbert \
        --test_files test1_text_only.csv \
        --output_dir ./predictions \
        --ensemble
"""

import argparse
import os
import re
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── Text cleaning (mirrors data_prep.py) ──────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).replace('\n', ' ')
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Dataset (ClinicalBERT inference) ──────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts     = texts
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
        }


# ── BERT inference ─────────────────────────────────────────────────────────────

def predict_bert(model, loader, device):
    """Returns hard predictions (0/1) and softmax probabilities."""
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds          = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
    return np.array(all_preds), np.array(all_probs)


# ── LR inference ──────────────────────────────────────────────────────────────

def predict_lr(pipeline, texts):
    """Returns hard predictions (0/1) and predict_proba output."""
    preds = pipeline.predict(texts)
    probs = pipeline.predict_proba(texts)
    return np.array(preds), np.array(probs)


# ── Ensemble ───────────────────────────────────────────────────────────────────

def ensemble_predict(bert_probs, lr_probs, bert_weight=0.7, lr_weight=0.3):
    """Weighted average of class probabilities; argmax for final label."""
    combined = bert_weight * bert_probs + lr_weight * lr_probs
    return np.argmax(combined, axis=1)


# ── Load models ────────────────────────────────────────────────────────────────

def load_bert(model_dir, device, hf_cache=None):
    bert_path = os.path.join(model_dir, 'best_model')
    if not os.path.isdir(bert_path):
        return None, None
    print(f"Loading ClinicalBERT from {bert_path}")
    tokenizer = AutoTokenizer.from_pretrained(bert_path, cache_dir=hf_cache)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    bert_path, cache_dir=hf_cache)
    model.to(device)
    return tokenizer, model


def load_lr(model_dir):
    lr_path = os.path.join(model_dir, 'lr_model', 'pipeline.pkl')
    if not os.path.isfile(lr_path):
        return None
    print(f"Loading LR pipeline from {lr_path}")
    with open(lr_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',   required=True,
                        help='Parent dir containing best_model/ and/or lr_model/.')
    parser.add_argument('--test_files',  nargs='+', required=True)
    parser.add_argument('--output_dir',  default='./predictions')
    parser.add_argument('--max_len',     type=int, default=128)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--use_lr_only', action='store_true',
                        help='Use only the Logistic Regression model.')
    parser.add_argument('--ensemble',    action='store_true',
                        help='Ensemble BERT + LR predictions (weighted average).')
    parser.add_argument('--bert_weight', type=float, default=0.7,
                        help='BERT weight in ensemble (LR weight = 1 - bert_weight).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hf_cache = os.environ.get('TRANSFORMERS_CACHE', None)
    print(f"Using device: {device}")

    # ── Load models ────────────────────────────────────────────────────────────
    tokenizer, bert_model = (None, None)
    lr_pipeline           = None

    if not args.use_lr_only:
        tokenizer, bert_model = load_bert(args.model_dir, device, hf_cache)
        if bert_model is None:
            print("  [WARN] No ClinicalBERT checkpoint found at "
                  f"{os.path.join(args.model_dir, 'best_model')}.")

    lr_pipeline = load_lr(args.model_dir)
    if lr_pipeline is None:
        print("  [WARN] No LR pipeline found at "
              f"{os.path.join(args.model_dir, 'lr_model', 'pipeline.pkl')}.")

    if bert_model is None and lr_pipeline is None:
        raise RuntimeError("No models found. Check --model_dir.")

    # Decide prediction mode
    use_bert     = (bert_model is not None) and (not args.use_lr_only)
    use_lr       = (lr_pipeline is not None)
    do_ensemble  = args.ensemble and use_bert and use_lr

    if do_ensemble:
        print(f"Mode: ENSEMBLE (BERT weight={args.bert_weight:.2f}, "
              f"LR weight={1-args.bert_weight:.2f})")
    elif use_bert:
        print("Mode: ClinicalBERT only")
    else:
        print("Mode: Logistic Regression only")

    # ── Process each test file ─────────────────────────────────────────────────
    for test_path in args.test_files:
        if not os.path.exists(test_path):
            print(f"  [WARN] Not found, skipping: {test_path}")
            continue

        df = pd.read_csv(test_path)
        df.columns = df.columns.str.strip().str.lower()

        text_col = next(
            (c for c in ('text', 'sentence', 'note', 'notes') if c in df.columns), None)
        if text_col is None:
            raise ValueError(f"No text column in {test_path}. Columns: {list(df.columns)}")

        id_col = next((c for c in ('id', 'idx', 'index') if c in df.columns), None)
        texts  = df[text_col].fillna('').apply(clean_text).tolist()

        # Logistic Regression predictions
        lr_preds, lr_probs = None, None
        if use_lr:
            lr_preds, lr_probs = predict_lr(lr_pipeline, texts)

        # ClinicalBERT predictions
        bert_preds, bert_probs = None, None
        if use_bert:
            dataset      = InferenceDataset(texts, tokenizer, args.max_len)
            loader       = DataLoader(dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=2, pin_memory=True)
            bert_preds, bert_probs = predict_bert(bert_model, loader, device)

        # Combine
        if do_ensemble:
            preds = ensemble_predict(bert_probs, lr_probs,
                                     bert_weight=args.bert_weight,
                                     lr_weight=1.0 - args.bert_weight)
        elif use_bert:
            preds = bert_preds
        else:
            preds = lr_preds

        # Build output dataframe
        if id_col:
            out_df = pd.DataFrame({'id': df[id_col].values, 'label': preds})
        else:
            out_df = pd.DataFrame({'id': range(len(preds)), 'label': preds})

        # Derive output filename: test1_text_only.csv → test1-pred.csv
        base     = os.path.basename(test_path)
        stem     = os.path.splitext(base)[0]
        stem     = re.sub(r'_text_only$', '', stem, flags=re.IGNORECASE)
        out_name = f"{stem}-pred.csv"
        out_path = os.path.join(args.output_dir, out_name)

        out_df.to_csv(out_path, index=False)
        pos_pct = preds.mean() * 100
        print(f"  {base} → {out_path}  "
              f"({len(preds)} rows, {pos_pct:.1f}% predicted positive)")

    print("\nPredictions complete.")


if __name__ == '__main__':
    main()
