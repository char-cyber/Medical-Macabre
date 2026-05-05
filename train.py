import argparse
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from preprocess import split_sentences, weak_label, clean_text
from model_utils import MODEL_NAMES, BertEmbedder, train_lr_on_embeddings, save_ensemble, label_to_binary

TEXT_CANDIDATES = ["sentence", "text", "note", "note_text", "TEXT", "clinical_note"]
LABEL_CANDIDATES = ["label", "useful", "target", "y", "codable"]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_labeled_csv(path):
    df = pd.read_csv(path)
    text_col = pick_col(df, TEXT_CANDIDATES)
    label_col = pick_col(df, LABEL_CANDIDATES)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text and label columns in {path}. Columns: {list(df.columns)}")
    out = pd.DataFrame({"sentence": df[text_col].astype(str).map(clean_text), "label": df[label_col].map(label_to_binary)})
    return out[out["sentence"].str.len() > 0]


def load_manual(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["sentence", "label"])
    df = pd.read_csv(path)
    text_col = pick_col(df, TEXT_CANDIDATES)
    label_col = pick_col(df, ["manual_label", "label", "useful", "target"])
    if text_col is None or label_col is None:
        raise ValueError(f"Manual labels need sentence/text and label columns. Columns: {list(df.columns)}")
    labels = df[label_col].apply(lambda x: 0 if str(x).strip() == "-1" else label_to_binary(x))
    return pd.DataFrame({"sentence": df[text_col].astype(str).map(clean_text), "label": labels})


def make_weak_examples(notes_glob, max_examples):
    rows = []
    if not notes_glob:
        return pd.DataFrame(columns=["sentence", "label"])
    for path in glob.glob(notes_glob):
        df = pd.read_csv(path)
        text_col = pick_col(df, TEXT_CANDIDATES)
        if text_col is None:
            continue
        for note in df[text_col].dropna().astype(str):
            for item in split_sentences(note):
                wl = weak_label(item["sentence"])
                # Treat -1 as a strong negative, because negated findings usually are not codable for the diagnosis itself.
                y = 0 if wl == -1 else wl
                rows.append({"sentence": item["sentence"], "label": y})
                if len(rows) >= max_examples:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--manual_csv", default="")
    ap.add_argument("--weak_notes_glob", default="")
    ap.add_argument("--weak_max", type=int, default=3000)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--cache_dir", default="./hf_cache")
    ap.add_argument("--out", default="artifacts/icd_ensemble.joblib")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    train = load_labeled_csv(args.train_csv)
    manual = load_manual(args.manual_csv)
    weak = make_weak_examples(args.weak_notes_glob, args.weak_max)

    df = pd.concat([train, manual, weak], ignore_index=True).drop_duplicates("sentence")
    df = df[df["sentence"].str.split().str.len().between(3, 128)]
    print(f"Loaded labeled rows: train={len(train)}, manual={len(manual)}, weak={len(weak)}, final={len(df)}")
    print(df["label"].value_counts())

    texts = df["sentence"].tolist()
    y = df["label"].astype(int).values
    strat = y if len(set(y)) > 1 and min(np.bincount(y)) >= 2 else None
    tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=strat)

    ensemble = {"models": [], "threshold": 0.5, "model_names": args.models}
    val_probs = []

    for model_name in args.models:
        print(f"\nEncoding with {model_name}")
        emb = BertEmbedder(model_name, cache_dir=args.cache_dir)
        X = emb.encode(texts, batch_size=args.batch_size)
        clf = train_lr_on_embeddings(X[tr_idx], y[tr_idx])
        p = clf.predict_proba(X[va_idx])[:, 1]
        val_probs.append(p)
        pred = (p >= 0.5).astype(int)
        print(classification_report(y[va_idx], pred, digits=4))
        ensemble["models"].append({"model_name": model_name, "clf": clf})

    avg = np.mean(np.vstack(val_probs), axis=0)
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.25, 0.75, 51):
        f1 = f1_score(y[va_idx], (avg >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    ensemble["threshold"] = best_t
    print(f"Best ensemble threshold={best_t:.2f}, val_f1={best_f1:.4f}")
    save_ensemble(args.out, ensemble)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
