import argparse
import os
import glob
import pandas as pd
import numpy as np
from preprocess import clean_text, split_sentences, weak_label
from model_utils import BertEmbedder, load_ensemble

TEXT_CANDIDATES = ["sentence", "text", "note", "note_text", "TEXT", "clinical_note"]
ID_CANDIDATES = ["id", "ID", "row_id", "sentence_id"]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_test_df(path):
    df = pd.read_csv(path)
    text_col = pick_col(df, TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError(f"Could not find text column in {path}. Columns: {list(df.columns)}")
    id_col = pick_col(df, ID_CANDIDATES)

    # If rows are full notes, split them. If rows are already sentences, this keeps one output per row.
    rows = []
    for idx, row in df.iterrows():
        raw = str(row[text_col])
        words = raw.split()
        if len(words) > 140 or "\n" in raw:
            pieces = split_sentences(raw)
            # For note level rows, aggregate later by original row id.
            for j, p in enumerate(pieces):
                rows.append({
                    "orig_index": idx,
                    "orig_id": row[id_col] if id_col else idx,
                    "piece_index": j,
                    "sentence": p["sentence"],
                })
        else:
            rows.append({
                "orig_index": idx,
                "orig_id": row[id_col] if id_col else idx,
                "piece_index": 0,
                "sentence": clean_text(raw),
            })
    return pd.DataFrame(rows), id_col


def predict_file(path, ensemble, cache_dir, batch_size, output_label_col):
    test, id_col = prepare_test_df(path)
    texts = test["sentence"].fillna("").astype(str).tolist()
    probs = []
    for item in ensemble["models"]:
        emb = BertEmbedder(item["model_name"], cache_dir=cache_dir)
        X = emb.encode(texts, batch_size=batch_size)
        probs.append(item["clf"].predict_proba(X)[:, 1])
    avg = np.mean(np.vstack(probs), axis=0)

    # Rule based safety adjustment. Negated clinical mentions get pushed down, strong clinical phrases get a tiny boost.
    adjusted = []
    for sent, p in zip(texts, avg):
        wl = weak_label(sent)
        if wl == -1:
            p = min(p, 0.35)
        elif wl == 1:
            p = max(p, p + 0.03)
        adjusted.append(p)
    test["probability"] = adjusted
    test[output_label_col] = (test["probability"] >= ensemble.get("threshold", 0.5)).astype(int)

    # If original file had full notes, use max over sentence chunks. If already sentence level, no change.
    out = test.groupby("orig_id", as_index=False).agg({output_label_col: "max", "probability": "max"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/icd_ensemble.joblib")
    ap.add_argument("--test_glob", required=True)
    ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--cache_dir", default="./hf_cache")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--output_label_col", default="prediction")
    ap.add_argument("--include_probability", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ensemble = load_ensemble(args.model)
    for path in sorted(glob.glob(args.test_glob)):
        out = predict_file(path, ensemble, args.cache_dir, args.batch_size, args.output_label_col)
        if not args.include_probability:
            out = out.drop(columns=["probability"], errors="ignore")
        base = os.path.basename(path).replace("_text_only.csv", "-pred.csv").replace(".csv", "-pred.csv")
        out_path = os.path.join(args.out_dir, base)
        out.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
