"""
data_augmentation.py — load external labelled data to augment training

Sources tried (in order):
  1. HuggingFace `mtsamuel/mimic_iii_icd_sentences`  (if internet available)
  2. Local MIMIC-III CSV if present in model_cache
  3. MedNLI / i2b2 style fallback patterns

All external samples are normalised to columns: ['text', 'label', 'label_3class', 'label_binary']
"""

import pandas as pd
import numpy as np
from pathlib import Path
from label_engineer import engineer_labels


# ---------------------------------------------------------------------------
# MIMIC-style heuristic labeler for unlabelled external sentences
# Used when we have raw MIMIC text but no gold labels.
# ---------------------------------------------------------------------------
import re
import os

USEFUL_KEYWORDS = re.compile(
    r"\b(diagnosis|diagnosed|presents?\s+with|history\s+of|"
    r"admitted\s+(for|with)|complaint\s+of|assessment|"
    r"impression|findings?|results?|procedure|surgery|"
    r"icd|code|dx|treatment|prescribed|medication)\b",
    re.IGNORECASE,
)

def heuristic_label(text: str) -> int:
    """
    Assign a weak heuristic label to unlabelled MIMIC sentences.
    Returns 1 if text likely contains ICD-codable info, else 0.
    This is intentionally conservative to avoid noise.
    """
    if USEFUL_KEYWORDS.search(text):
        return 1
    return 0


def load_from_huggingface(cache_dir: Path) -> pd.DataFrame | None:
    """
    Attempt to load a HuggingFace dataset.
    Grace compute nodes usually cannot access internet, so skip when offline.
    """
    if os.environ.get("HF_DATASETS_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        print("  Offline mode enabled — skipping HuggingFace external data")
        return None
    try:
        from datasets import load_dataset
        print("  Trying HuggingFace: 'bigbio/mednli'...")
        # MedNLI: sentence pairs with entailment labels.
        # We use "entailment" pairs as useful (1) and "contradiction" as negated (-1).
        ds = load_dataset("bigbio/mednli", trust_remote_code=True,
                          cache_dir=str(cache_dir))
        records = []
        for split in ["train", "validation"]:
            for ex in ds[split]:
                # sentence1 is always a clinical observation → label 1
                records.append({"text": ex["sentence1"], "label": 1})
                # sentence2 label depends on relation
                lbl = 1 if ex["label"] == "entailment" else 0
                records.append({"text": ex["sentence2"], "label": lbl})
        df = pd.DataFrame(records).drop_duplicates(subset="text")
        df = df[df["text"].str.split().str.len() >= 4]
        print(f"  MedNLI loaded: {len(df)} rows")
        return df
    except Exception as e:
        print(f"  HuggingFace load failed ({e}) — skipping")
        return None


def load_from_local_mimic(cache_dir: Path) -> pd.DataFrame | None:
    """
    If the user has placed a local MIMIC CSV (raw sentences, no labels)
    at cache_dir/mimic_sentences.csv, apply heuristic labels.
    """
    local_path = cache_dir / "mimic_sentences.csv"
    if not local_path.exists():
        print("  No local mimic_sentences.csv found — skipping")
        return None

    df = pd.read_csv(local_path)
    if "text" not in df.columns:
        print("  mimic_sentences.csv missing 'text' column — skipping")
        return None

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.split().str.len() >= 4]

    if "label" not in df.columns:
        print("  Applying heuristic labels to local MIMIC data...")
        df["label"] = df["text"].apply(heuristic_label)

    print(f"  Local MIMIC loaded: {len(df)} rows")
    return df


def load_external_data(cache_dir: Path) -> pd.DataFrame | None:
    """
    Try all external sources and return a combined, normalised DataFrame
    with columns: ['text', 'label', 'label_3class', 'label_binary']
    Returns None if nothing could be loaded.
    """
    dfs = []

    hf_df = load_from_huggingface(cache_dir)
    if hf_df is not None:
        dfs.append(hf_df)

    mimic_df = load_from_local_mimic(cache_dir)
    if mimic_df is not None:
        dfs.append(mimic_df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="text")
    combined["label"] = combined["label"].astype(int)

    # Apply 3-class label engineering (adds negation detection)
    combined = engineer_labels(combined, label_col="label")

    # Cap external data to avoid drowning the small gold set
    MAX_EXTERNAL = 500
    if len(combined) > MAX_EXTERNAL:
        # Stratified sample to preserve balance
        combined = combined.groupby("label_binary", group_keys=False).apply(
            lambda g: g.sample(min(len(g), MAX_EXTERNAL), random_state=42)
        ).reset_index(drop=True)
        print(f"  External data capped to {len(combined)} rows (stratified)")

    return combined
