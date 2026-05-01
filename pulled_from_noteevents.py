import pandas as pd
import os
import nltk
import random
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

path = r'C:\Users\siyab\OneDrive\Desktop\CSCE421\final'
# important label1 keywords 
label_1_keywords = [
    "diagnosis", "diagnosed", "history of", "hx of",
    "acute", "chronic", "disease", "failure",
    "infection", "pneumonia", "sepsis",
    "diabetes", "hypertension", "cancer",
    "fracture", "cardiac", "respiratory",
    "stroke", "anemia", "pain", "bleeding",
    "treated", "admitted", "procedure",
    "surgery", "medication", "antibiotic"
]
#important label0 keywords
label_0_keywords = [
    "please call", "follow up", "follow-up",
    "family updated", "appointment",
    "insurance", "social work",
    "discharge instructions", "thank you",
    "signed", "dictated by",
    "no issues", "stable condition"
]

def build_dataset():

    notes = pd.read_csv(os.path.join(path, 'NOTEEVENTS.csv'))
    diagnoses = pd.read_csv(os.path.join(path, 'DIAGNOSES_ICD.csv'))
    # makes sure hadm_ids are unique and drop the rows with no hadm_id
    icd_hadm_ids = set(diagnoses['HADM_ID'].dropna().unique())
    #seperate notes with an icd anf notes without an icd code 
    notes_with_icd = notes[notes['HADM_ID'].isin(icd_hadm_ids)]
    notes_without_icd = notes[~notes['HADM_ID'].isin(icd_hadm_ids)]

    # drop notes that have an empty text
    notes_with_icd = notes_with_icd.dropna(subset=['TEXT'])
    notes_without_icd = notes_without_icd.dropna(subset=['TEXT'])

    # Sample 100 each
    sample_with_icd = notes_with_icd.sample(100, random_state=42)
    sample_without_icd = notes_without_icd.sample(100, random_state=42)

    combined_notes = pd.concat([sample_with_icd, sample_without_icd])


    label1 = []
    label0 = []
    #look at the text column
    for text in combined_notes['TEXT']:
        clean = str(text).replace('\n', ' ')
        #split notes into sentences and sentences into words 
        for s in sent_tokenize(clean):
            words = s.split()
            # makes sure the length of each sentence is longer than 20 words and no longer than 128 
            if 20 <= len(words) <= 128:
                s_lower = s.lower()
                
                has_label1 = any(k in s_lower for k in label_1_keywords)
                has_label0 = any(k in s_lower for k in label_0_keywords)
                # makes sure any label1 sentences not have a word in label0
                # if true add it to label1
                if has_label1 and not has_label0:
                    label1.append(s.strip())

                elif not has_label1:
                    label0.append(s.strip())

    print(f"Label1 candidates: {len(label1)}")
    print(f"Label0 candidates: {len(label0)}")

    # Shuffle
    random.shuffle(label1)
    random.shuffle(label0)

    # Balance
    target = min(len(label1), len(label0))

    label1 = label1[:target]
    label0 = label0[:target]

    data = []

    for s in label1:
        data.append({"text": s, "label": 1})

    for s in label0:
        data.append({"text": s, "label": 0})

    df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)

    print("Final distribution:")
    print(df['label'].value_counts())
    print(f"Total rows: {len(df)}")

    output_file = os.path.join(path, 'final_sentence_dataset.csv')
    df.to_csv(output_file, index=False)

    print("Saved dataset!")

if __name__ == "__main__":
    build_dataset()