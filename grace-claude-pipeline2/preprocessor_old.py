"""
preprocessor.py — section-split with regex, then sentence-split with scispaCy

Flow for each row:
  raw text
    → section_split()   : identify known clinical sections (HPI, PMH, Assessment…)
    → sentence_split()  : scispaCy en_core_sci_sm tokenizes each section into sentences
    → clean_sentence()  : normalise whitespace, lowercase optional

The dataframe is exploded so that one output row = one sentence.
The original row index is preserved as `source_row` for traceability.
"""

import re
import pandas as pd

# ── scispaCy (lazy-loaded so import errors surface clearly) ──────────────────
_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_sci_sm")
            # Disable components we don't need — only need sentencizer
            _NLP.disable_pipes([p for p in _NLP.pipe_names
                                 if p not in ("tok2vec", "sentencizer", "senter")])
        except OSError:
            raise OSError(
                "scispaCy model not found. Run:\n"
                "  pip install scispacy\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/"
                "releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
            )
    return _NLP


# =============================================================================
# SECTION HEADERS — clinical note sections we want to identify and split on
# Add more patterns here if your corpus uses non-standard headings.
# =============================================================================
SECTION_PATTERNS = [
    # Common structured sections
    r"(?i)\b(chief\s+complaint|cc)\s*:",
    r"(?i)\b(history\s+of\s+present\s+illness|hpi)\s*:",
    r"(?i)\b(past\s+(medical\s+)?history|pmh?|past\s+surgical\s+history|psh)\s*:",
    r"(?i)\b(review\s+of\s+systems?|ros)\s*:",
    r"(?i)\b(medications?|meds?|current\s+medications?)\s*:",
    r"(?i)\b(allergies?)\s*:",
    r"(?i)\b(physical\s+exam(ination)?|pe|vitals?)\s*:",
    r"(?i)\b(assessment(\s+and\s+plan)?|a/?p|impression)\s*:",
    r"(?i)\b(plan)\s*:",
    r"(?i)\b(diagnosis|diagnoses|dx)\s*:",
    r"(?i)\b(lab(oratory)?\s+(results?|data|values?)?)\s*:",
    r"(?i)\b(imaging|radiology)\s*:",
    r"(?i)\b(procedures?)\s*:",
    r"(?i)\b(discharge\s+(summary|instructions?|diagnosis|medications?))\s*:",
    r"(?i)\b(family\s+history|fh?x?)\s*:",
    r"(?i)\b(social\s+history|sh?x?)\s*:",
    r"(?i)\b(findings?)\s*:",
    r"(?i)\b(impression\s+and\s+recommendation)\s*:",
    r"(?i)\b(follow[\-\s]?up)\s*:",
]

# Combined pattern that anchors to start of a line or after a newline
_SECTION_RE = re.compile(
    r"(?:^|\n)(" + "|".join(f"(?:{p})" for p in SECTION_PATTERNS) + r")",
    re.MULTILINE,
)


def section_split(text: str) -> list[dict]:
    """
    Split a clinical note into sections.

    Returns a list of dicts:
        [{"header": "HPI:", "body": "Patient is a 45-year-old..."}, ...]

    If no section headers are found, returns a single entry with header=None.
    """
    text = text.strip()
    matches = list(_SECTION_RE.finditer(text))

    if not matches:
        return [{"header": None, "body": text}]

    sections = []
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start  = m.end()
        end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body   = text[start:end].strip()
        if body:
            sections.append({"header": header, "body": body})

    # Text before the first header (e.g. patient name line)
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.insert(0, {"header": None, "body": preamble})

    return sections


def sentence_split(text: str) -> list[str]:
    """
    Use scispaCy to split a block of text into sentences.
    Falls back to a naive period-split if scispaCy is unavailable.
    """
    try:
        nlp = _get_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        # Fallback: naive split on ". " boundaries
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]


def clean_sentence(sent: str) -> str:
    """Normalise whitespace; keep original case (BERT is cased)."""
    sent = re.sub(r"[\n\r\t]+", " ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    return sent.strip()


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Explode each row's text into individual sentences via section → sentence split.

    Input df columns : text_col, [label], [any other columns]
    Output df columns: text (sentence), section_header, source_row,
                       + all original columns except text_col
    """
    records = []
    label_col_present = "label" in df.columns

    for row_idx, row in df.iterrows():
        raw_text = str(row[text_col])
        label    = row["label"] if label_col_present else None

        sections = section_split(raw_text)

        for sec in sections:
            sentences = sentence_split(sec["body"])
            for sent in sentences:
                sent_clean = clean_sentence(sent)
                if len(sent_clean.split()) < 3:
                    # Skip very short fragments (e.g. lone numbers, headers)
                    continue
                rec = {
                    "text":           sent_clean,
                    "section_header": sec["header"],
                    "source_row":     row_idx,
                }
                if label_col_present:
                    rec["label"] = label
                records.append(rec)

    out = pd.DataFrame(records)
    print(f"  Sentences after split: {len(out)}  (from {len(df)} original rows)")
    return out.reset_index(drop=True)
