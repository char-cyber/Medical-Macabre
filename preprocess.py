import re
from typing import List, Tuple

SECTION_RE = re.compile(
    r"(?im)^\s*(admission diagnosis|discharge diagnosis|chief complaint|history of present illness|past medical history|hospital course|assessment and plan|assessment|plan|medications|allergies|physical exam|labs?|imaging|procedures?|disposition|follow up|diagnosis|impression)\s*:?\s*$"
)
NEGATION_RE = re.compile(
    r"\b(no|not|without|denies|denied|negative for|no evidence of|ruled out|rule out|absence of|free of)\b",
    re.IGNORECASE,
)
CLINICAL_SIGNAL_RE = re.compile(
    r"\b(diagnosis|diagnoses|disease|syndrome|failure|fracture|infection|pneumonia|sepsis|diabetes|hypertension|asthma|copd|cancer|tumou?r|lesion|ischemia|infarct|bleed|hemorrhage|renal|hepatic|cardiac|respiratory|acute|chronic|history of|status post|s/p|treated|started|continued|elevated|low|abnormal|positive)\b",
    re.IGNORECASE,
)
ADMIN_RE = re.compile(
    r"\b(date|signed|dictated|attending|provider|phone|fax|page|mrn|name|unit no|admission date|discharge date)\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)  # MIMIC de-id placeholders
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sections(note: str) -> List[Tuple[str, str]]:
    """Return (section_name, section_text) pairs. Falls back to one UNKNOWN section."""
    raw = note or ""
    matches = list(SECTION_RE.finditer(raw))
    if not matches:
        return [("UNKNOWN", raw)]
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        name = clean_text(m.group(1)).upper()
        body = raw[start:end].strip()
        if body:
            sections.append((name, body))
    return sections or [("UNKNOWN", raw)]


def get_nlp():
    try:
        import spacy
        try:
            return spacy.load("en_core_sci_sm")
        except Exception:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp
    except Exception:
        return None


def regex_sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|\s{2,}", text)
    return [clean_text(p) for p in parts if len(clean_text(p)) > 2]


def split_sentences(note: str, max_words: int = 128) -> List[dict]:
    nlp = get_nlp()
    rows = []
    for section, body in split_sections(note):
        body = body.replace(";", ". ")
        if nlp is not None:
            try:
                sents = [clean_text(s.text) for s in nlp(body).sents]
            except Exception:
                sents = regex_sentence_split(body)
        else:
            sents = regex_sentence_split(body)
        for sent in sents:
            words = sent.split()
            if not words:
                continue
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                if len(chunk.split()) >= 3:
                    rows.append({"section": section, "sentence": chunk})
    return rows


def weak_label(sentence: str) -> int:
    """Optional weak label. 1 means likely codable, 0 means likely not codable, -1 means negated clinical mention."""
    s = clean_text(sentence)
    if NEGATION_RE.search(s) and CLINICAL_SIGNAL_RE.search(s):
        return -1
    if CLINICAL_SIGNAL_RE.search(s) and not ADMIN_RE.search(s):
        return 1
    return 0
