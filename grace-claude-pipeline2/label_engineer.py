"""
label_engineer.py — extends binary labels to 3-class labels

Classes:
   1  = useful        (contains ICD-codable info)
   0  = not useful    (generic / no medical info)
  -1  = negated       (explicitly negated finding: "no evidence of X", "denies X")

Strategy:
  • Start from binary label column (1/0).
  • For any sentence labelled 1 (useful), check for strong negation patterns.
    If negation is detected, downgrade to -1.
  • For downstream binary prediction (Gradescope expects 0/1), map:
      -1 → 0  (negated = not codable in the positive sense)
       0 → 0
       1 → 1
"""

import re
import pandas as pd

# =============================================================================
# NEGATION PATTERNS
# These cover the most common clinical negation constructions.
# Add more if you find false negatives in your data.
# =============================================================================
NEGATION_PATTERNS = [
    # "no evidence of", "no signs of", "no history of"
    r"\bno\s+(evidence|signs?|history|findings?|indication|mention|known|documented|prior|previous|acute|active|significant)\s+(of\s+)?",
    # "denies", "denied"
    r"\bdenies?\b",
    # "without", "w/o"
    r"\bwithout\b",
    r"\bw/o\b",
    # "negative for"
    r"\bnegative\s+for\b",
    # "ruled out", "rule out"
    r"\brule[d]?\s+out\b",
    # "not present", "not found", "not seen", "not reported"
    r"\bnot\s+(present|found|seen|reported|detected|identified|noted|observed|significant|demonstrated)\b",
    # "absent", "absence of"
    r"\babsence\s+of\b",
    r"\babsent\b",
    # "unremarkable", "within normal limits", "WNL"
    r"\bunremarkable\b",
    r"\bwithin\s+normal\s+limits?\b",
    r"\b(wnl)\b",
    # "clear of", "free of", "free from"
    r"\b(clear|free)\s+(of|from)\b",
    # "no complaints of"
    r"\bno\s+complaints?\s+(of\s+)?",
    # "never had", "never diagnosed"
    r"\bnever\s+(had|diagnosed|experienced|reported)\b",
]

_NEG_RE = re.compile(
    "|".join(NEGATION_PATTERNS),
    re.IGNORECASE,
)


def has_negation(text: str) -> bool:
    """Return True if text contains a strong negation marker."""
    return bool(_NEG_RE.search(text))


def engineer_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Add a `label_3class` column:
      original 1 + no negation  →  1
      original 1 + negation     → -1
      original 0                →  0

    Also adds:
      `has_negation`  : bool, whether sentence triggered a negation pattern
      `label_binary`  : final 0/1 label for Gradescope submission
                        (-1 maps to 0)
    """
    labels_3class = []
    has_neg_flags = []

    for _, row in df.iterrows():
        orig_label = int(row[label_col])
        neg        = has_negation(str(row["text"]))
        has_neg_flags.append(neg)

        if orig_label == 1 and neg:
            labels_3class.append(-1)
        else:
            labels_3class.append(orig_label)

    df = df.copy()
    df["has_negation"] = has_neg_flags
    df["label_3class"] = labels_3class

    # Binary label for submission: -1 → 0
    df["label_binary"] = df["label_3class"].apply(lambda x: 1 if x == 1 else 0)

    neg_count = (df["label_3class"] == -1).sum()
    print(f"  Negation-downgraded sentences : {neg_count}")
    print(f"  Final 3-class distribution:\n{df['label_3class'].value_counts().to_string()}")
    return df
