"""
predictor.py — runs ensemble inference on each test file and writes prediction CSVs

Output format per Gradescope spec:
  testX-pred.csv
  Columns: id, label
    id    : row index (0-based) matching original test CSV row order
    label : 0 or 1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from trainer import Ensemble

# NEGATION_PATTERNS = [
#     "no evidence of", "denies", "negative for",
#     "without evidence of", "unremarkable", "no signs of"
# ]

# def apply_negation_penalty(text, prob):
#     t = text.lower()
#     for p in NEGATION_PATTERNS:
#         if p in t:
#             return prob * 0.6
#     return prob

# added
import re

STRONG_KEYWORDS = [
    "pneumonia",
    "sepsis",
    "diabetes",
    "hypertension",
    "heart failure",
    "renal failure",
    "kidney failure",
    "copd",
    "asthma",
    "anemia",
    "fracture",
    "infection",
    "cancer",
    "tumor",
    "embolism",
    "thrombosis",
    "hemorrhage",
    "ischemia",
]

NEGATION_PHRASES = [
    "no evidence of",
    "no signs of",
    "negative for",
    "denies",
    "without evidence of",
    "without signs of",
    "rule out",
    "ruled out",
    "r/o",
    "no",
    "not",
]

HISTORY_PHRASES = [
    "history of",
    "hx of",
    "family history of",
    "prior history of",
]

WINDOW_SIZE = 6


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def has_history(sentence):
    text = str(sentence).lower()
    return any(p in text for p in HISTORY_PHRASES)


def keyword_is_negated(tokens, keyword_start):
    start = max(0, keyword_start - WINDOW_SIZE)
    context = " ".join(tokens[start:keyword_start])

    return any(phrase in context for phrase in NEGATION_PHRASES)


def contains_non_negated_keyword(sentence):
    tokens = tokenize(sentence)

    for keyword in STRONG_KEYWORDS:
        keyword_tokens = tokenize(keyword)
        k = len(keyword_tokens)

        for i in range(len(tokens) - k + 1):
            if tokens[i:i+k] == keyword_tokens:
                if keyword_is_negated(tokens, i):
                    return False
                if has_history(sentence):
                    return False
                return True

    return False


def apply_heuristic(text, prob, threshold):
    """
    Returns final sentence label, final probability, and reason.
    """

    bert_label = int(prob >= threshold)

    # Case 1: BERT says 0, but strong keyword says likely ICD-relevant
    if bert_label == 0 and contains_non_negated_keyword(text):
        return 1, max(prob, 0.99), "keyword_override"

    # Case 2: BERT says 1, but sentence is clearly negated
    if bert_label == 1:
        tokens = tokenize(text)
        for keyword in STRONG_KEYWORDS:
            keyword_tokens = tokenize(keyword)
            k = len(keyword_tokens)

            for i in range(len(tokens) - k + 1):
                if tokens[i:i+k] == keyword_tokens and keyword_is_negated(tokens, i):
                    return 0, min(prob, 0.01), "negation_override"

    return bert_label, prob, "bert_prediction"
# -----------------------------------------------------------------------------------------

def predict_test_files(
    test_dfs: dict,
    ensemble: Ensemble,
    output_dir: Path,
    device,
):
    """
    Parameters
    ----------
    test_dfs   : dict { 'test1': df, ... } — each df has column 'text'
                 (already preprocessed; source_row tracks original row index)
    ensemble   : trained Ensemble object
    output_dir : where to write testX-pred.csv files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # threshold = getattr(ensemble, "threshold", 0.5)
    threshold = 0.20
    print(f"  Using prediction threshold: {threshold:.3f}")

    for test_name, df in test_dfs.items():
        print(f"\n  Predicting {test_name} ({len(df)} rows)")

        df = df.copy()
        texts = df["text"].tolist()

        # Get P(class=1), NOT hard 0/1 predictions
        probs = ensemble.predict_proba_class1(texts)

        # Apply negation suppression
        # df["prob"] = [
        #     apply_negation_penalty(text, prob)
        #     for text, prob in zip(df["text"], probs)
        # ]

        # APPLY HUERISTIC 
        heuristic_results = [
            apply_heuristic(text, prob, threshold)
            for text, prob in zip(df["text"], probs)
        ]

        df["sentence_label"] = [x[0] for x in heuristic_results]
        df["prob"] = [x[1] for x in heuristic_results]
        df["override_reason"] = [x[2] for x in heuristic_results]


        # Aggregate sentence probabilities back to original row
        # if "source_row" in df.columns:
        #     row_preds = (
        #         df.groupby("source_row")["prob"]
        #         .mean()
        #         .reset_index()
        #         .rename(columns={"source_row": "id"})
        #     )
        # else:
        #     row_preds = pd.DataFrame({
        #         "id": range(len(df)),
        #         "prob": df["prob"].values,
        #     })

        # CHANGED sentence probs 
        if "source_row" in df.columns:
            row_preds = (
                df.groupby("source_row")
                .agg(
                    prob=("prob", "mean"),
                    label=("sentence_label", "max"),
                    percent_1_sentences=("sentence_label", "mean"),
                )
                .reset_index()
                .rename(columns={"source_row": "id"})
            )

            row_preds["percent_0_sentences"] = 1 - row_preds["percent_1_sentences"]

        else:
            row_preds = pd.DataFrame({
                "id": range(len(df)),
                "prob": df["prob"].values,
                "label": df["sentence_label"].values,
                "percent_1_sentences": df["sentence_label"].values,
            })
            row_preds["percent_0_sentences"] = 1 - row_preds["percent_1_sentences"]

        # Apply calibrated threshold
        row_preds["label"] = (row_preds["prob"] >= threshold).astype(int)

        # Gradescope only wants id,label
        # row_preds = row_preds[["id", "label"]]
        row_preds = row_preds[["id", "label", "prob"]]

        row_preds["id"] = row_preds["id"].astype(int)
        row_preds["label"] = row_preds["label"].astype(int)

        row_preds = row_preds.sort_values("id").reset_index(drop=True)

        out_path = output_dir / f"{test_name}-pred.csv"
        row_preds.to_csv(out_path, index=False)

        print(f"  Written → {out_path} ({len(row_preds)} rows)")
        print(f"    Label dist: {row_preds['label'].value_counts().to_dict()}")
