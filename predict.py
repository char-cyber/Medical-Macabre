"""
predict.py
==========
Each row in the test CSV is a full clinical note (row_id, text).
Pipeline:
  1. Split each note into sentences  (section regex + scispaCy, same as data_prep.py)
  2. Predict every sentence          (ClinicalBERT and/or Logistic Regression)
  3. Aggregate back to note level    (fraction of sentences labeled 1 >= threshold)
  4. Write one output row per note   (row_id, label)

Usage:
    # BERT only (default)
    python predict.py \
        --model_dir  /scratch/user/charu7465/baseline_results/clinicalbert \
        --test_files test01_text_only.csv test02_text_only.csv test03_text_only.csv \
        --output_dir ./predictions \
        --threshold  0.3

    # LR only
    python predict.py ... --use_lr_only

    # Ensemble
    python predict.py ... --ensemble --bert_weight 0.7
"""

import argparse
import os
import re
import pickle
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Section splitting  (identical to data_prep.py)
# ─────────────────────────────────────────────────────────────────────────────

SECTION_PATTERNS = [
    r"(?i)^(chief complaint|cc)\s*:",
    r"(?i)^(history of present illness|hpi)\s*:",
    r"(?i)^(past medical history|pmh|past medical hx)\s*:",
    r"(?i)^(past surgical history|psh)\s*:",
    r"(?i)^(medications?|current medications?|meds)\s*:",
    r"(?i)^(allergies|nkda|nkma)\s*:",
    r"(?i)^(review of systems?|ros)\s*:",
    r"(?i)^(physical exam(ination)?|pe|vital signs?|vitals?)\s*:",
    r"(?i)^(labs?|laboratory|lab results?|laboratory data)\s*:",
    r"(?i)^(imaging|radiology|radiologic)\s*:",
    r"(?i)^(assessment|impression|findings?)\s*:",
    r"(?i)^(assessment and plan|a/p|a&p)\s*:",
    r"(?i)^(plan|treatment plan)\s*:",
    r"(?i)^(diagnosis|diagnoses|dx)\s*:",
    r"(?i)^(discharge (diagnosis|summary|instructions?|condition|medications?))\s*:",
    r"(?i)^(procedures?|operative note)\s*:",
    r"(?i)^(social history|sh)\s*:",
    r"(?i)^(family history|fh)\s*:",
    r"(?i)^(hospital course|brief hospital course)\s*:",
    r"(?i)^(follow.?up)\s*:",
]
_SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.MULTILINE)


def split_into_sections(note_text):
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


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting  (identical to data_prep.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_spacy():
    try:
        import spacy
        try:
            return spacy.load("en_core_sci_sm")
        except OSError:
            try:
                return spacy.load("en_core_sci_lg")
            except OSError:
                print("  [WARN] scispaCy model not found -- falling back to sentencizer.")
                nlp = spacy.blank("en")
                nlp.add_pipe("sentencizer")
                return nlp
    except ImportError:
        print("  [WARN] spaCy not installed -- using regex sentence splitter.")
        return None


def regex_sentence_split(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_sentences(text, nlp):
    if nlp is None:
        return regex_sentence_split(text)
    MAX_CHARS = 100_000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    return sents if sents else regex_sentence_split(text)


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning  (identical to data_prep.py)
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text):
    text = str(text).replace('\n', ' ')
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def note_to_sentences(note_text, nlp):
    sentences = []
    for section in split_into_sections(note_text):
        for sent in split_sentences(section, nlp):
            s = clean_text(sent)
            if len(s) >= 15:
                sentences.append(s)
    if not sentences:
        s = clean_text(note_text)
        if s:
            sentences = [s]
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-level inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_bert_sentences(texts, tokenizer, model, device, batch_size, max_len):
    if not texts:
        return np.array([], dtype=int), np.zeros((0, 2))
    dataset = InferenceDataset(texts, tokenizer, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds          = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
    return np.array(all_preds), np.array(all_probs)


def predict_lr_sentences(texts, pipeline):
    if not texts:
        return np.array([], dtype=int), np.zeros((0, 2))
    preds = pipeline.predict(texts)
    probs = pipeline.predict_proba(texts)
    return np.array(preds), np.array(probs)


# ─────────────────────────────────────────────────────────────────────────────
# Note-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_note(sent_preds, threshold):
    if len(sent_preds) == 0:
        return 0
    return int(sent_preds.mean() >= threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_bert(model_dir, device, hf_cache=None):
    bert_path = os.path.join(model_dir, 'best_model')
    if not os.path.isdir(bert_path):
        return None, None
    print(f"Loading ClinicalBERT from {bert_path}")
    tokenizer = AutoTokenizer.from_pretrained(bert_path, cache_dir=hf_cache)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    bert_path, cache_dir=hf_cache)
    model.to(device)
    return tokenizer, model


def load_lr(model_dir):
    lr_path = os.path.join(model_dir, 'lr_model', 'pipeline.pkl')
    if not os.path.isfile(lr_path):
        return None
    print(f"Loading LR pipeline from {lr_path}")
    with open(lr_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',   required=True)
    parser.add_argument('--test_files',  nargs='+', required=True)
    parser.add_argument('--output_dir',  default='./predictions')
    parser.add_argument('--max_len',     type=int,   default=128)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--threshold',   type=float, default=0.3,
                        help='Fraction of sentences predicted 1 needed to label a note '
                             'as 1. E.g. 0.3 means >=30%% of sentences must be positive.')
    parser.add_argument('--use_lr_only', action='store_true')
    parser.add_argument('--ensemble',    action='store_true')
    parser.add_argument('--bert_weight', type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hf_cache = os.environ.get('TRANSFORMERS_CACHE', None)
    print(f"Using device: {device}")
    print(f"Threshold: {args.threshold} "
          f"(note=1 if >= {args.threshold*100:.0f}% of its sentences are predicted 1)")

    # Load scispaCy
    print("Loading scispaCy...")
    nlp = load_spacy()

    # Load models
    tokenizer, bert_model = None, None
    lr_pipeline = None

    if not args.use_lr_only:
        tokenizer, bert_model = load_bert(args.model_dir, device, hf_cache)
        if bert_model is None:
            print(f"  [WARN] No ClinicalBERT checkpoint at "
                  f"{os.path.join(args.model_dir, 'best_model')}.")

    lr_pipeline = load_lr(args.model_dir)

    if bert_model is None and lr_pipeline is None:
        raise RuntimeError("No models found. Check --model_dir.")

    use_bert    = (bert_model is not None) and (not args.use_lr_only)
    use_lr      = (lr_pipeline is not None)
    do_ensemble = args.ensemble and use_bert and use_lr

    if do_ensemble:
        print(f"Mode: ENSEMBLE  (BERT {args.bert_weight:.0%} / LR {1-args.bert_weight:.0%})")
    elif use_bert:
        print("Mode: ClinicalBERT only")
    else:
        print("Mode: Logistic Regression only")

    # Process each test file
    for test_path in args.test_files:
        if not os.path.exists(test_path):
            print(f"  [WARN] Not found, skipping: {test_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {test_path}")
        print(f"{'='*60}")

        df = pd.read_csv(test_path)
        df.columns = df.columns.str.strip().str.lower()

        id_col   = next((c for c in ('row_id', 'id', 'idx', 'index') if c in df.columns), None)
        text_col = next((c for c in ('text', 'sentence', 'note', 'notes') if c in df.columns), None)
        if text_col is None:
            raise ValueError(f"No text column in {test_path}. Columns: {list(df.columns)}")

        note_ids   = df[id_col].tolist() if id_col else list(range(len(df)))
        note_texts = df[text_col].fillna('').tolist()

        note_labels      = []
        note_sent_counts = []
        note_pos_fracs   = []

        for i, (note_id, note_text) in enumerate(zip(note_ids, note_texts)):
            sentences = note_to_sentences(note_text, nlp)

            if not sentences:
                note_labels.append(0)
                note_sent_counts.append(0)
                note_pos_fracs.append(0.0)
                continue

            # Sentence-level predictions
            if do_ensemble:
                bert_preds, bert_probs = predict_bert_sentences(
                    sentences, tokenizer, bert_model, device,
                    args.batch_size, args.max_len)
                lr_preds, lr_probs = predict_lr_sentences(sentences, lr_pipeline)
                combined   = args.bert_weight * bert_probs + (1 - args.bert_weight) * lr_probs
                sent_preds = np.argmax(combined, axis=1)
            elif use_bert:
                sent_preds, _ = predict_bert_sentences(
                    sentences, tokenizer, bert_model, device,
                    args.batch_size, args.max_len)
            else:
                sent_preds, _ = predict_lr_sentences(sentences, lr_pipeline)

            pos_frac   = float(sent_preds.mean())
            note_label = aggregate_note(sent_preds, args.threshold)

            note_labels.append(note_label)
            note_sent_counts.append(len(sentences))
            note_pos_fracs.append(pos_frac)
            print(f"    note {note_id}: {len(sentences)} sentences, {pos_frac*100:.1f}% positive → label={note_label}")


            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(note_ids)} notes...")

        # Output filename: test01_text_only.csv -> test01-pred.csv
        base     = os.path.basename(test_path)
        stem     = os.path.splitext(base)[0]
        stem     = re.sub(r'_text_only$', '', stem, flags=re.IGNORECASE)
        out_name = f"{stem}-pred.csv"
        out_path = os.path.join(args.output_dir, out_name)

        # Gradescope submission file (id + label only)
        pd.DataFrame({'id': note_ids, 'label': note_labels}).to_csv(out_path, index=False)

        # Debug file with sentence stats
        debug_path = out_path.replace('.csv', '_debug.csv')
        pd.DataFrame({
            'id':            note_ids,
            'label':         note_labels,
            'n_sentences':   note_sent_counts,
            'pct_sent_pos':  [f"{f*100:.1f}%" for f in note_pos_fracs],
        }).to_csv(debug_path, index=False)

        # Summary
        n_notes    = len(note_labels)
        n_positive = sum(note_labels)
        avg_sents  = np.mean(note_sent_counts) if note_sent_counts else 0
        avg_pos    = np.mean(note_pos_fracs)   if note_pos_fracs   else 0

        print(f"\n  Results for {base}:")
        print(f"    Total notes:          {n_notes}")
        print(f"    Labeled 1 (codable):  {n_positive}  ({n_positive/n_notes*100:.1f}%)")
        print(f"    Labeled 0:            {n_notes-n_positive}  ({(n_notes-n_positive)/n_notes*100:.1f}%)")
        print(f"    Avg sentences/note:   {avg_sents:.1f}")
        print(f"    Avg % sentences pos:  {avg_pos*100:.1f}%")
        print(f"    Threshold used:       {args.threshold}")
        print(f"    Submission file:      {out_path}")
        print(f"    Debug file:           {debug_path}")

    print("\nPredictions complete.")


if __name__ == '__main__':
    main()