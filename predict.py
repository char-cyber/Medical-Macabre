"""
predict.py
Runs inference with the saved ClinicalBERT checkpoint on one or more test CSV
files and writes  testXX-pred.csv  files that the Gradescope autograder expects.

Each output CSV has two columns: id (or the original index) and label (0 or 1).

Usage:
    python predict.py \
        --model_dir  /scratch/user/charu7465/baseline_results/clinicalbert/best_model \
        --test_files test1_text_only.csv test2_text_only.csv \
        --output_dir ./predictions \
        --max_len 128 \
        --batch_size 32
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── Dataset (no labels) ───────────────────────────────────────────────────────

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


# ── Text cleaning (mirrors data_prep.py) ─────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).replace('\n', ' ')
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',   required=True,
                        help='Path to the saved ClinicalBERT checkpoint (best_model/).')
    parser.add_argument('--test_files',  nargs='+', required=True,
                        help='One or more test CSV files.')
    parser.add_argument('--output_dir',  default='./predictions')
    parser.add_argument('--max_len',     type=int, default=128)
    parser.add_argument('--batch_size',  type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load model once ───────────────────────────────────────────────────────
    hf_cache = os.environ.get('TRANSFORMERS_CACHE', None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, cache_dir=hf_cache)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    args.model_dir, cache_dir=hf_cache)
    model.to(device)
    print(f"Model loaded from {args.model_dir}")

    # ── Process each test file ────────────────────────────────────────────────
    for test_path in args.test_files:
        if not os.path.exists(test_path):
            print(f"  [WARN] Not found, skipping: {test_path}")
            continue

        df = pd.read_csv(test_path)
        df.columns = df.columns.str.strip().str.lower()

        # Accept 'text', 'sentence', or 'note' as the text column
        text_col = next((c for c in ('text', 'sentence', 'note', 'notes')
                         if c in df.columns), None)
        if text_col is None:
            raise ValueError(f"Cannot find a text column in {test_path}. "
                             f"Columns: {list(df.columns)}")

        # Detect id column (or fall back to row index)
        id_col = next((c for c in ('id', 'idx', 'index') if c in df.columns), None)

        texts = df[text_col].fillna('').apply(clean_text).tolist()

        dataset = InferenceDataset(texts, tokenizer, args.max_len)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

        preds = predict(model, loader, device)

        # Build output dataframe
        if id_col:
            out_df = pd.DataFrame({'id': df[id_col].values, 'label': preds})
        else:
            out_df = pd.DataFrame({'id': range(len(preds)), 'label': preds})

        # Derive output filename: test1_text_only.csv → test1-pred.csv
        base = os.path.basename(test_path)
        stem = os.path.splitext(base)[0]                    # test1_text_only
        stem = re.sub(r'_text_only$', '', stem, flags=re.IGNORECASE)  # test1
        out_name = f"{stem}-pred.csv"
        out_path = os.path.join(args.output_dir, out_name)

        out_df.to_csv(out_path, index=False)
        pos_pct = preds.mean() * 100
        print(f"  {base} → {out_path}  "
              f"({len(preds)} rows, {pos_pct:.1f}% predicted positive)")

    print("\nPredictions complete.")


if __name__ == '__main__':
    main()
