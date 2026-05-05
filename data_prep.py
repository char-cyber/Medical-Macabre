"""
data_prep.py
Merges all labeled CSV files, applies the weak-labeling heuristic to uncertain
rows, shuffles, and writes train.csv / val.csv ready for ClinicalBERT fine-tuning.

Usage (run locally or on Grace login node — no GPU needed):
    python data_prep.py \
        --inputs train_data-text_and_labels.csv additional_data.csv labeled_add_data.csv \
        --output_dir ./prepared_data
"""

import argparse
import os
import re
import random
import pandas as pd
import numpy as np

# ── Weak-label heuristic (from label.py) ────────────────────────────────────

DIAGNOSIS_PATTERNS = [
    r'\bdiagnosed with\b', r'\bdx of\b', r'\bimpression of\b',
    r'\bconsistent with\b', r'\bconfirmed\b',
]
CONDITION_KEYWORDS = [
    r'diabetes', r'hypertension', r'cancer', r'tumor', r'mass',
    r'infection', r'pneumonia', r'sepsis', r'heart failure',
    r'kidney failure', r'liver failure', r'syndrome', r'disorder',
    r'injury', r'fracture', r'copd', r'asthma', r'anemia',
    r'stroke', r'myocardial infarction', r'cva', r'tia',
    r'depression', r'anxiety', r'thyroid',
]
PROCEDURE_PATTERNS = [
    r'\bunderwent\b', r'\bperformed\b', r'\bsurgery\b', r'\bbiopsy\b',
    r'\bresection\b', r'\bdialysis\b', r'\bintubation\b', r'\bventilation\b',
]
LAB_PATTERNS = [
    r'\belevated\b', r'\bdecreased\b', r'\bpositive\b', r'\bnegative\b',
    r'\bwhite blood cell\b', r'\bhemoglobin\b', r'\bplatelet\b',
    r'\bblood pressure\b', r'\bheart rate\b',
]
SYMPTOM_PATTERNS = [
    r'\bcomplains of\b', r'\bpresented with\b', r'\breports\b', r'\bc/o\b',
    r'\bpain\b', r'\bfever\b', r'\bcough\b', r'\bnausea\b',
    r'\bshortness of breath\b', r'\bfatigue\b',
]
NON_RELEVANT_PATTERNS = [
    r'\bfollow up\b', r'\bappointment\b', r'\bpatient instructed\b',
    r'\bdiscussed with\b', r'\bwill continue\b', r'\bstable\b',
    r'\bdischarge\b', r'\badmission date\b', r'\bdate of birth\b',
    r'\bmale\b', r'\bfemale\b', r'\blives with\b', r'\bresides\b',
    r'no evidence', r'unremarkable', r'unclear', r'unknown',
    r'unchanged', r'not consistent',
]


def weak_label(text: str, original_label: int) -> int:
    """Re-score a sentence; return 1/0.  Ties defer to original_label."""
    txt = str(text).lower()
    score = 0
    for p in DIAGNOSIS_PATTERNS:
        if re.search(p, txt): score += 5
    for p in CONDITION_KEYWORDS:
        if re.search(p, txt): score += 3
    for p in PROCEDURE_PATTERNS:
        if re.search(p, txt): score += 4
    for p in LAB_PATTERNS:
        if re.search(p, txt): score += 2
    for p in SYMPTOM_PATTERNS:
        if re.search(p, txt): score += 1
    for p in NON_RELEVANT_PATTERNS:
        if re.search(p, txt): score -= 3

    if score >= 3:
        return 1
    elif score <= -2:
        return 0
    return original_label


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    # normalise column names: accept 'text'/'sentence'/'note' and 'label'/'labels'
    rename = {}
    for col in df.columns:
        if col in ('sentence', 'note', 'notes'):
            rename[col] = 'text'
        if col in ('labels', 'target'):
            rename[col] = 'label'
    df = df.rename(columns=rename)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"{path} must have 'text' and 'label' columns. Found: {list(df.columns)}")
    df = df[['text', 'label']].dropna(subset=['text'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce').dropna().astype(int)
    df = df.dropna(subset=['label'])
    return df


def clean_text(text: str) -> str:
    text = str(text).replace('\n', ' ')
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True,
                        help='One or more labeled CSV files to merge.')
    parser.add_argument('--output_dir', default='./prepared_data',
                        help='Where to write train.csv and val.csv.')
    parser.add_argument('--val_frac', type=float, default=0.15,
                        help='Fraction held out for validation (default 0.15).')
    parser.add_argument('--apply_weak_label', action='store_true',
                        help='Re-score every row with the heuristic.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load & merge
    frames = []
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"  [WARN] File not found, skipping: {path}")
            continue
        df = load_csv(path)
        print(f"  Loaded {len(df):,} rows from {path}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid input files found.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined):,} rows before dedup")

    # 2. Clean
    combined['text'] = combined['text'].apply(clean_text)
    combined = combined[combined['text'].str.len() > 10]

    # 3. Deduplicate on text
    combined = combined.drop_duplicates(subset=['text'], keep='first')
    print(f"After dedup:  {len(combined):,} rows")

    # 4. Optional weak-label pass
    if args.apply_weak_label:
        print("Applying weak-label heuristic...")
        combined['label'] = combined.apply(
            lambda r: weak_label(r['text'], int(r['label'])), axis=1
        )

    # 5. Class balance report
    vc = combined['label'].value_counts()
    print(f"\nClass distribution:\n{vc.to_string()}")
    pos_count = vc.get(1, 0)
    neg_count = vc.get(0, 0)
    ratio = pos_count / max(neg_count, 1)
    print(f"Pos/Neg ratio: {ratio:.3f}")
    if ratio < 0.3 or ratio > 3.0:
        print("  [WARN] Class imbalance detected. Consider oversampling positives or using class weights.")

    # 6. Shuffle & split
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(combined) * args.val_frac)
    val_df = combined.iloc[:n_val]
    train_df = combined.iloc[n_val:]

    # 7. Save
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path   = os.path.join(args.output_dir, 'val.csv')
    train_df[['text', 'label']].to_csv(train_path, index=False)
    val_df[['text', 'label']].to_csv(val_path,   index=False)

    print(f"\nSaved {len(train_df):,} train rows  → {train_path}")
    print(f"Saved {len(val_df):,}   val rows    → {val_path}")
    print("Done.")


if __name__ == '__main__':
    main()
