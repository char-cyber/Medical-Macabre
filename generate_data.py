import pandas as pd
import re

POS_HEADERS = ["discharge diagnosis", "final diagnosis", "diagnoses", "impression", "assessment"]

NEG_PAT = re.compile(
    r"\b(no|denies|denied|without|negative for|normal|unremarkable|unchanged|wnl|did not|not show|no evidence)\b",
    re.I
)

VITAL_PAT = re.compile(r"\b(bp|hr|temp|mmhg|pulse|resp|spo2)\b", re.I)

DISEASE_PAT = re.compile(
    r"\b(pneumonia|sepsis|bacteremia|cardiomyopathy|effusion|fracture|infection|syndrome|disease|cva|cad|anemia)\b",
    re.I
)

# 🔥 STRONGER TREATMENT FILTER
TREATMENT_PAT = re.compile(
    r"\b(started|continued|given|treated|received|course|dose|mg|tablet|iv|po|infusion|therapy)\b",
    re.I
)
# ADD THIS NEAR YOUR PATTERNS
HISTORY_PAT = re.compile(r"\b(history of|h/o)\b", re.I)

ADMIN_PAT = re.compile(r"\b(admitted|transferred|discharged|placed|scheduled|followup|rehab)\b", re.I)

PLAN_PAT = re.compile(r"\b(should|will|plan|evaluate|re-evaluate|follow|consider)\b", re.I)

FAMILY_PAT = re.compile(r"\b(mother|father|family history|sister|brother)\b", re.I)

PROCEDURE_PAT = re.compile(r"\b(sent|sample|test|study|scan|imaging|evaluation|reveals?|showed?)\b", re.I)
WEAK_POS_PAT = re.compile(
    r"\b(inflammation|infection|effusion|edema|thickening|lesion|opacity|ascites|disease)\b",
    re.I
)
POSTOP_PAT = re.compile(r"\b(status post|s/p|underwent)\b", re.I)
CARDIAC_PAT = re.compile(
    r"\b(atrial fibrillation|afib|tachycardia|bradycardia|st depression|st elevation|arrhythmia)\b",
    re.I
)
HEADER_PAT = re.compile(r"\b(cxr|ct|mri|u/s)\b", re.I)
IMAGING_POS_PAT = re.compile(
    r"\b(echo|ct|mri|cxr).*(effusion|edema|lesion|thrombus|aneurysm|fracture)\b",
    re.I
)
HISTORY_PAT = re.compile(r"\b(history of|h/o)\b", re.I)
def clean(s):
    return " ".join(str(s).replace("\n", " ").split())

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def label_sentence(sent, header):
    s = sent.lower()
    h = header.lower() if header else ""

    # -------- HARD NEGATIVES --------
    if NEG_PAT.search(s):
        return 0

    if VITAL_PAT.search(s) and len(s.split()) < 6:
        return 0

    if HEADER_PAT.search(s) and len(s.split()) < 5:
        return 0

    # -------- STRONG POSITIVES (CHECK FIRST) --------
    if IMAGING_POS_PAT.search(s):
        return 1
    if CARDIAC_PAT.search(s):
        return 1

    if any(p in h for p in POS_HEADERS):
        return 1

    if DISEASE_PAT.search(s) and not NEG_PAT.search(s):
        return 1

    if WEAK_POS_PAT.search(s) and not NEG_PAT.search(s):
        if len(s.split()) > 4:
            return 1

    # -------- NOW FILTER NON-DIAGNOSTIC --------
    if (
        TREATMENT_PAT.search(s)
        or ADMIN_PAT.search(s)
        or PLAN_PAT.search(s)
        or FAMILY_PAT.search(s)
        or POSTOP_PAT.search(s)
        or HISTORY_PAT.search(s)
    ):
        return 0

    return None


def extract_clean_dataset(notes_file, n_notes=8000, max_per_class=3000):
    df = pd.read_csv(notes_file, nrows=n_notes)

    texts, labels = [], []

    for _, row in df.iterrows():
        raw_text = str(row.get("TEXT", ""))

        sections = re.split(r"\n([A-Za-z ]+):", raw_text)

        for i in range(1, len(sections), 2):
            header = sections[i]
            content = sections[i + 1]

            for sent in split_sentences(content):
                sent = clean(sent)
                if not sent:
                    continue

                words = sent.split()

                if len(words) < 3:
                    continue

                if len(words) > 128:
                    sent = " ".join(words[:128])

                label = label_sentence(sent, header)

                if label is None:
                    continue

                texts.append(sent)
                labels.append(label)

    df_all = pd.DataFrame({"text": texts, "label": labels}).drop_duplicates("text")

    if len(df_all) == 0:
        print("❌ No samples extracted.")
        return df_all

    df0 = df_all[df_all.label == 0]
    df1 = df_all[df_all.label == 1]

    if len(df0) == 0 or len(df1) == 0:
        print("❌ One class empty")
        print(df_all.label.value_counts())
        return df_all

    n = min(len(df0), len(df1), max_per_class)

    df_out = pd.concat([
        df0.sample(n, random_state=42),
        df1.sample(n, random_state=42)
    ]).sample(frac=1, random_state=42)

    df_out.to_csv("new_train.csv", index=False)

    print(f"\nSaved {len(df_out)} samples")
    print(df_out.label.value_counts())

    return df_out

if __name__ == "__main__":
    extract_clean_dataset("NOTEEVENTS.csv", n_notes=5000, max_per_class=2000)