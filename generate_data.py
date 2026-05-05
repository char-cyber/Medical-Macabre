"""Generate calibrated training data: 200 positive / 500 negative (~29% positive)
to match the estimated true positive rate of test01/test02 (~28-30%).

Strategy:
  1. Read MIMIC-III NOTEEVENTS.csv.
  2. Split each note into (section_header, body) pairs.
  3. For each section body, split into sentences.
  4. Label each sentence based on:
       - Strong-positive sections: discharge diagnosis, final diagnosis,
         diagnoses, impression, assessment, plan, hospital course summary.
       - Strong-negative sections: social history, family history,
         administrative metadata (admission date, name, MRN, etc.),
         vitals snapshots.
       - Contextual sections (HPI, exam, labs): require an ICD keyword
         match to be labeled positive.
  5. Apply negation / normal-finding overrides to flip false positives.
  6. Sample to 200 pos / 500 neg.
  7. Concatenate the 20 gold examples from train_data-text_and_labels.csv.

Output: data/combined/combined_train.csv with columns text,label.

Env vars:
  NOTES_CSV    : path to NOTEEVENTS.csv (default data/NOTEEVENTS.csv)
  N_NOTES      : how many notes to scan (default 8000)
  WANT_POS     : positive samples to keep (default 200)
  WANT_NEG     : negative samples to keep (default 500)
  GOLD_CSV     : path to gold labels (default data/train_data-text_and_labels.csv)
  OUT_DIR      : output dir (default data/combined)
  SEED         : RNG seed (default 42)
"""

import os
import re
import pandas as pd

# ---------------------------------------------------------------- patterns ---
POS_HEADERS = [
    "discharge diagnosis", "discharge diagnoses",
    "final diagnosis", "final diagnoses",
    "diagnoses", "diagnosis",
    "impression",
    "assessment", "assessment and plan",
    "hospital course",
    "active issues", "problem list",
    "principal diagnosis",
]

NEG_HEADERS = [
    "social history",
    "family history",
    "allergies",
    "admission date", "discharge date",
    "name", "unit no", "service", "attending", "dictated by",
    "addendum",
]

CONTEXTUAL_HEADERS = [
    "history of present illness", "hpi",
    "physical exam", "physical examination", "exam",
    "labs", "laboratory", "labs on admission",
    "imaging", "studies",
    "review of systems", "ros",
    "medications on admission", "medications on discharge",
    "chief complaint",
]

# Common ICD-codable disease/symptom keywords. Intentionally narrow so
# borderline matches stay unlabeled rather than mislabeled.
ICD_KEYWORDS = re.compile(
    r"\b("
    r"pneumonia|sepsis|septic|bacteremia|cellulitis|abscess|"
    r"diabet|hypertens|hyperlipid|ckd|renal failure|aki|esrd|"
    r"copd|asthma|emphysema|pulmonary embol|pe|dvt|"
    r"stroke|cva|tia|seizure|epilepsy|"
    r"myocardial|infarct|mi|cad|chf|heart failure|cardiomyopath|"
    r"af|atrial fib|atrial flutt|svt|vt|arrhythmia|"
    r"cancer|tumou?r|carcinoma|lymphoma|leukemia|malignan|metasta|"
    r"fracture|hemorrhage|bleed|ich|sah|sdh|"
    r"hepatitis|cirrhosis|liver failure|"
    r"anemia|thrombocytopenia|leukocytosis|"
    r"depression|anxiety|psychos|schizophren|bipolar|"
    r"copd|gerd|ulcer|gastritis|colitis|pancreatitis|cholecystitis|"
    r"uti|urinary tract infection|pyelonephritis|"
    r"hypothyroid|hyperthyroid|"
    r"obesity|malnutrition|"
    r"osteoporosis|osteoarthritis|rheumatoid"
    r")\b", re.IGNORECASE,
)

NORMAL_FINDING = re.compile(
    r"\b("
    r"no acute (cardiopulmonary|process|change|finding|distress)"
    r"|within normal limits|wnl"
    r"|unremarkable|nontender|nondistended"
    r"|no new|no interval change"
    r"|normal sinus rhythm|nsr"
    r"|negative for|no evidence of"
    r"|denies|denied"
    r")\b", re.IGNORECASE,
)

ADMIN_PAT = re.compile(
    r"\b(date|signed|dictated|attending|provider|phone|fax|page|mrn|"
    r"unit no|admission date|discharge date|cc:|copy to)\b", re.IGNORECASE,
)

VITALS_PAT = re.compile(
    r"\b(bp|hr|temp|rr|spo2|sat|o2 sat|pulse|resp rate|mmhg|"
    r"\d+\s*/\s*\d+|\d+\s*bpm)\b", re.IGNORECASE,
)


# ----------------------------------------------------------------- helpers ---
def clean(s):
    return " ".join(str(s).replace("\n", " ").split())


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def split_sections(note):
    note = note or ""
    parts = re.split(r"\n([A-Za-z][A-Za-z /\-]{2,40}):", note)
    if len(parts) < 3:
        yield "", note
        return
    yield "", parts[0]
    for i in range(1, len(parts), 2):
        header = parts[i].strip().lower()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        yield header, body


def header_class(header):
    h = header.lower()
    for p in POS_HEADERS:
        if p in h:
            return "pos"
    for p in NEG_HEADERS:
        if p in h:
            return "neg"
    for p in CONTEXTUAL_HEADERS:
        if p in h:
            return "ctx"
    return "other"


def label_sentence(sent, header):
    s = sent.strip()
    if len(s.split()) < 3:
        return None
    if NORMAL_FINDING.search(s):
        return 0
    if ADMIN_PAT.search(s) and not ICD_KEYWORDS.search(s):
        return 0
    if VITALS_PAT.search(s) and not ICD_KEYWORDS.search(s):
        return 0
    cls = header_class(header)
    if cls == "pos":
        return 1
    if cls == "neg":
        return 0
    if cls == "ctx":
        return 1 if ICD_KEYWORDS.search(s) else 0
    if ICD_KEYWORDS.search(s):
        return 1
    return None


# ------------------------------------------------------------------ main -----
def extract(notes_csv, n_notes, want_pos, want_neg, seed=42):
    print(f"Reading {notes_csv} (first {n_notes} notes)")
    df = pd.read_csv(notes_csv, nrows=n_notes)
    text_col = "TEXT" if "TEXT" in df.columns else df.columns[-1]
    pos_rows, neg_rows = [], []
    for _, row in df.iterrows():
        note = str(row.get(text_col, "")).replace("[**", " ").replace("**]", " ")
        for header, body in split_sections(note):
            for sent in split_sentences(body):
                sent = clean(sent)
                words = sent.split()
                if len(words) < 3:
                    continue
                if len(words) > 128:
                    sent = " ".join(words[-128:])
                lbl = label_sentence(sent, header)
                if lbl == 1:
                    pos_rows.append(sent)
                elif lbl == 0:
                    neg_rows.append(sent)
        if len(pos_rows) >= want_pos * 5 and len(neg_rows) >= want_neg * 5:
            break

    pos_df = pd.DataFrame({"text": pos_rows, "label": 1}).drop_duplicates("text")
    neg_df = pd.DataFrame({"text": neg_rows, "label": 0}).drop_duplicates("text")
    print(f"Candidate pool: {len(pos_df)} positive, {len(neg_df)} negative")

    pos_df = pos_df.sample(n=min(want_pos, len(pos_df)), random_state=seed)
    neg_df = neg_df.sample(n=min(want_neg, len(neg_df)), random_state=seed)
    return pd.concat([pos_df, neg_df], ignore_index=True)


def main():
    notes_path = os.environ.get("NOTES_CSV", "data/NOTEEVENTS.csv")
    n_notes = int(os.environ.get("N_NOTES", "8000"))
    want_pos = int(os.environ.get("WANT_POS", "200"))
    want_neg = int(os.environ.get("WANT_NEG", "500"))
    gold_csv = os.environ.get("GOLD_CSV", "data/train_data-text_and_labels.csv")
    out_dir = os.environ.get("OUT_DIR", "data/combined")
    seed = int(os.environ.get("SEED", "42"))

    sampled = extract(notes_path, n_notes, want_pos, want_neg, seed=seed)
    print(f"Sampled augment: {sampled['label'].value_counts().to_dict()}")

    if os.path.exists(gold_csv):
        gold = pd.read_csv(gold_csv)
        if "text" in gold.columns and "label" in gold.columns:
            gold = gold[["text", "label"]].copy()
            gold["label"] = gold["label"].astype(int).clip(0, 1)
            sampled = pd.concat([gold, sampled], ignore_index=True)
            print(f"After adding {len(gold)} gold examples: {sampled['label'].value_counts().to_dict()}")

    sampled = sampled.drop_duplicates("text").reset_index(drop=True)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "combined_train.csv")
    sampled.to_csv(out_path, index=False)
    print(f"Saved {len(sampled)} rows to {out_path}")
    print(f"Final balance: {sampled['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
