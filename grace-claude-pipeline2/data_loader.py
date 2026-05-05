"""
data_loader.py — loads train CSV, all test CSVs, and optional manual labels
"""

import pandas as pd
import glob
from pathlib import Path


def load_all_data(train_csv, test_dir, manual_csv=None):
    """
    Returns
    -------
    train_df : pd.DataFrame  with columns ['text', 'label']
    test_dfs : dict  { 'test1': df, 'test2': df, ... }  each with column ['text']
    """
    # ── Train ────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(train_csv)
    train_df = train_df.dropna(subset=["text"])
    train_df["text"] = train_df["text"].astype(str).str.strip()

    # Normalize label column: accept 'label', 'Label', 'useful', etc.
    if "label" not in train_df.columns:
        # try common aliases
        for alias in ["Label", "useful", "Useful", "y", "Y"]:
            if alias in train_df.columns:
                train_df = train_df.rename(columns={alias: "label"})
                break
    train_df["label"] = train_df["label"].astype(int)

    # ── Manual labels (751 sentences) ────────────────────────────────────────
    if manual_csv is not None and Path(manual_csv).exists():
        manual_df = pd.read_csv(manual_csv)
        manual_df = manual_df.dropna(subset=["text"])
        manual_df["text"] = manual_df["text"].astype(str).str.strip()
        if "label" not in manual_df.columns:
            for alias in ["Label", "useful", "y"]:
                if alias in manual_df.columns:
                    manual_df = manual_df.rename(columns={alias: "label"})
                    break
        manual_df["label"] = manual_df["label"].astype(int)
        train_df = pd.concat([train_df, manual_df], ignore_index=True)
        print(f"  Appended {len(manual_df)} manual-labeled rows → total {len(train_df)}")

    # ── Test files ───────────────────────────────────────────────────────────
    test_dir = Path(test_dir)
    test_dfs = {}
    for path in sorted(test_dir.glob("test*_text_only.csv")):
        name = path.stem.replace("_text_only", "")   # e.g. "test1"
        df = pd.read_csv(path)
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        test_dfs[name] = df
        print(f"  Loaded {name}: {len(df)} rows")

    return train_df, test_dfs
