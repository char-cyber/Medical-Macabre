# pip install pandas spacy
# python -m spacy download en_core_web_sm

import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load your dataset
df = pd.read_csv("train_data-text_and_labels.csv")

rows = []

for _, row in df.iterrows():
    note_id = row["row_id"]
    note_text = str(row["text"])
    note_label = row["label"]   # this is NOTE-level label

    doc = nlp(note_text)

    for sent in doc.sents:
        sentence = sent.text.strip()

        if sentence:
            rows.append({
                "note_id": note_id,
                "sentence": sentence,
                "note_label": note_label,
                "label": -1   # <-- placeholder for sentence label
            })

sentence_df = pd.DataFrame(rows)

sentence_df.to_csv("clinical_sentences_split.csv", index=False)

print("Done! Created sentence-level dataset.")