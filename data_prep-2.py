"""
data_prep.py - Sentence-level negative sampling
"""

import argparse
import os
import re
import random
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Section headers ─────────────────────────────────────────────────────────

SECTION_PATTERNS = [
    r"^(chief complaint|cc)\s*:",
    r"^(history of present illness|hpi)\s*:",
    r"^(past medical history|pmh|past medical hx)\s*:",
    r"^(past surgical history|psh)\s*:",
    r"^(medications?|current medications?|meds)\s*:",
    r"^(allergies|nkda|nkma)\s*:",
    r"^(review of systems?|ros)\s*:",
    r"^(physical exam(ination)?|pe|vital signs?|vitals?)\s*:",
    r"^(labs?|laboratory|lab results?|laboratory data)\s*:",
    r"^(imaging|radiology|radiologic)\s*:",
    r"^(assessment|impression|findings?)\s*:",
    r"^(assessment and plan|a/p|a&p)\s*:",
    r"^(plan|treatment plan)\s*:",
    r"^(diagnosis|diagnoses|dx)\s*:",
    r"^(discharge (diagnosis|summary|instructions?|condition|medications?))\s*:",
    r"^(procedures?|operative note)\s*:",
    r"^(social history|sh)\s*:",
    r"^(family history|fh)\s*:",
    r"^(hospital course|brief hospital course)\s*:",
    r"^(follow.?up)\s*:",
]
_SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.MULTILINE | re.IGNORECASE)

# ── Stronger sentence-level classifier ───────────────────────────────────────

# Strong positive indicators (diagnosis-related)
STRONG_POSITIVE = [
    r'\bdiagnosed with\b', r'\bdx of\b', r'\bimpression of\b',
    r'\bconsistent with\b', r'\bconfirmed\b', r'\bdifferential diagnosis\b',
    r'\badmitting diagnosis\b', r'\bdischarge diagnosis\b',
    r'\bprimary diagnosis\b', r'\bfinal diagnosis\b', r'\brule out\b',
]

# Disease keywords (positive)
DISEASE_KEYWORDS = [
    r'diabetes', r'hypertension', r'cancer', r'tumor', r'mass', r'neoplasm',
    r'infection', r'pneumonia', r'sepsis', r'heart failure', r'chf',
    r'kidney failure', r'liver failure', r'syndrome', r'disorder',
    r'injury', r'fracture', r'copd', r'asthma', r'anemia',
    r'stroke', r'myocardial infarction', r'cva', r'tia',
    r'depression', r'anxiety', r'thyroid', r'metastasis',
    r'hemorrhage', r'thrombosis', r'embolism', r'arrhythmia',
    r'pulmonary embolism', r'dvt', r'pneumothorax', r'effusion'
]

# Strong negative indicators (non-diagnosis sentences)
STRONG_NEGATIVE = [
    r'\bfollow up\b', r'\bappointment\b', r'\bpatient instructed\b',
    r'\bdiscussed with\b', r'\bwill continue\b', r'\bstable\b',
    r'\bdischarge\b', r'\badmission date\b', r'\bdate of birth\b',
    r'\bmale\b', r'\bfemale\b', r'\blives with\b', r'\bresides\b',
    r'no evidence', r'unremarkable', r'unclear', r'unknown',
    r'unchanged', r'not consistent', r'\bnormal\b', r'\bwithin normal limits\b',
    r'\bpatient denies\b', r'\bpatient reports no\b', r'\bnegative for\b',
    r'\bvital signs\b', r'\btemperature\b', r'\bpulse\b', r'\brespiratory rate\b',
    r'\bblood pressure\b', r'\bovernight\b', r'\badmitted to\b', r'\btransferred to\b',
    r'\bdischarged to\b', r'\bprimary care\b', r'\bfollow-up\b'
]

def classify_sentence(sentence: str) -> int:
    """Classify sentence as positive (1) or negative (0) based on content."""
    sent_lower = sentence.lower()
    
    # Check strong positives first
    for pattern in STRONG_POSITIVE:
        if re.search(pattern, sent_lower):
            return 1
    
    # Check strong negatives
    for pattern in STRONG_NEGATIVE:
        if re.search(pattern, sent_lower):
            return 0
    
    # Count disease keywords
    disease_count = sum(1 for pattern in DISEASE_KEYWORDS if re.search(pattern, sent_lower))
    
    # Heuristic: if multiple disease keywords, likely positive
    if disease_count >= 2:
        return 1
    elif disease_count == 1:
        # Ambiguous - check length (short sentences with disease might be positive)
        return 1 if len(sentence.split()) < 15 else 0
    else:
        return 0


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).replace('\n', ' ')
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Section splitting ─────────────────────────────────────────────────────────

def split_into_sections(note_text: str) -> list[str]:
    note_text = str(note_text)
    boundaries = [m.start() for m in _SECTION_RE.finditer(note_text)]
    if not boundaries:
        return [note_text]
    
    sections = []
    boundaries.append(len(note_text))
    for i in range(len(boundaries) - 1):
        chunk = note_text[boundaries[i]:boundaries[i + 1]]
        chunk = re.sub(r'^[^\n]+\n?', '', chunk, count=1)
        chunk = chunk.strip()
        if chunk:
            sections.append(chunk)
    return sections if sections else [note_text]


# ── Sentence splitting ───────────────────────────────────────────────────────

def load_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_sci_sm")
        except OSError:
            try:
                nlp = spacy.load("en_core_sci_lg")
            except OSError:
                print("  [WARN] scispaCy not found. Using regex sentence splitting.")
                nlp = None
    except ImportError:
        print("  [WARN] spaCy not installed. Using regex sentence splitting.")
        nlp = None
    return nlp

def regex_sentence_split(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def split_sentences(text: str, nlp) -> list[str]:
    if nlp is None:
        return regex_sentence_split(text)
    MAX_CHARS = 100_000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences if sentences else regex_sentence_split(text)


# ── Extract with sentence-level balancing ───────────────────────────────────

def extract_from_mimic_sentence_balanced(noteevents_path: str,
                                           diagnoses_path: str,
                                           nlp,
                                           max_sentences: int = 20000,
                                           chunksize: int = 500,
                                           seed: int = 42) -> pd.DataFrame:
    """
    Extract sentences and balance at the sentence level using heuristics.
    """
    random.seed(seed)
    target_per_class = max_sentences // 2
    
    print(f"\n[MIMIC] Loading DIAGNOSES_ICD...")
    diag_df = pd.read_csv(diagnoses_path, usecols=['HADM_ID'])
    diag_df.columns = diag_df.columns.str.lower()
    positive_ids = set(diag_df['hadm_id'].dropna().astype(int).tolist())
    print(f"  {len(positive_ids):,} HADM_IDs with diagnoses")
    
    # Stream notes and classify sentences
    pos_sentences = []
    neg_sentences = []
    
    chunk_iter = pd.read_csv(
        noteevents_path,
        usecols=['HADM_ID', 'TEXT'],
        chunksize=chunksize,
        low_memory=False,
    )
    
    chunks_read = 0
    print(f"[MIMIC] Extracting and classifying sentences...")
    
    for chunk in chunk_iter:
        chunk.columns = chunk.columns.str.lower()
        chunk = chunk.dropna(subset=['hadm_id', 'text'])
        chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce')
        chunk = chunk.dropna(subset=['hadm_id'])
        chunk['hadm_id'] = chunk['hadm_id'].astype(int)
        
        # Only use notes from patients with diagnoses (all of them)
        chunk = chunk[chunk['hadm_id'].isin(positive_ids)]
        
        for _, row in chunk.iterrows():
            sections = split_into_sections(row['text'])
            for section in sections:
                sentences = split_sentences(section, nlp)
                for sent in sentences[:30]:  # Limit per note to avoid overload
                    sent_clean = clean_text(sent)
                    if len(sent_clean) < 15 or len(sent_clean) > 500:
                        continue
                    
                    # Classify sentence
                    label = classify_sentence(sent_clean)
                    
                    if label == 1 and len(pos_sentences) < target_per_class:
                        pos_sentences.append({'text': sent_clean, 'label': 1, 'source': 'mimic'})
                    elif label == 0 and len(neg_sentences) < target_per_class:
                        neg_sentences.append({'text': sent_clean, 'label': 0, 'source': 'mimic'})
                    
                    # Early stop if both buckets full
                    if len(pos_sentences) >= target_per_class and len(neg_sentences) >= target_per_class:
                        break
            
            if len(pos_sentences) >= target_per_class and len(neg_sentences) >= target_per_class:
                break
        
        chunks_read += 1
        if chunks_read % 10 == 0:
            print(f"  ... chunk {chunks_read} | pos: {len(pos_sentences)}/{target_per_class} "
                  f"| neg: {len(neg_sentences)}/{target_per_class}")
        
        if len(pos_sentences) >= target_per_class and len(neg_sentences) >= target_per_class:
            print("  Both buckets full — stopping.")
            break
    
    print(f"  Collected {len(pos_sentences)} positive, {len(neg_sentences)} negative sentences")
    
    # Combine and shuffle
    all_sentences = pos_sentences + neg_sentences
    random.shuffle(all_sentences)
    
    return pd.DataFrame(all_sentences)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noteevents', default = 'NOTEEVENTS.csv')
    parser.add_argument('--diagnoses', default = 'DIAGNOSES_ICD.csv')
    parser.add_argument('--output_dir', default='./prepared_data')
    parser.add_argument('--max_sentences', type=int, default=20000,
                        help='Total sentences to extract (balanced)')
    parser.add_argument('--chunksize', type=int, default=500)
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[Setup] Loading sentence splitter...")
    nlp = load_spacy()
    
    # Extract with sentence-level balancing
    df = extract_from_mimic_sentence_balanced(
        noteevents_path=args.noteevents,
        diagnoses_path=args.diagnoses,
        nlp=nlp,
        max_sentences=args.max_sentences,
        chunksize=args.chunksize,
        seed=args.seed,
    )
    
    # Class distribution
    vc = df['label'].value_counts()
    print(f"\nFinal class distribution:\n{vc.to_string()}")
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(df) * args.val_frac)
    
    train_df = df.iloc[n_val:]
    val_df = df.iloc[:n_val]
    
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path = os.path.join(args.output_dir, 'val.csv')
    train_df[['text', 'label']].to_csv(train_path, index=False)
    val_df[['text', 'label']].to_csv(val_path, index=False)
    
    print(f"\nSaved {len(train_df):,} train rows → {train_path}")
    print(f"Saved {len(val_df):,} val rows → {val_path}")
    print("Done.")


if __name__ == '__main__':
    main()