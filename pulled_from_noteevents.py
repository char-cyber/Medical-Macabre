import pandas as pd
import nltk
import os
import random
from nltk.tokenize import sent_tokenize

# --- 1. Initial Setup & NLTK Downloads ---
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except:
    pass

path = r'C:\Users\siyab\OneDrive\Desktop\CSCE421\final'

def prepare_balanced_data():
    # Load your local MIMIC CSVs
    print("Loading local CSV files... (This may take a minute)")
    notes = pd.read_csv(os.path.join(path, 'NOTEEVENTS.csv'))
    diagnoses = pd.read_csv(os.path.join(path, 'DIAGNOSES_ICD.csv'))

    # Identify admissions that have ICD codes
    useful_hadms = diagnoses['HADM_ID'].unique()
    
    # Take the top 2500 notes for each category to extract sentences from
# Shuffling and assigning in one go
    pos_notes = notes[notes['HADM_ID'].isin(useful_hadms)].sample(frac=1, random_state=42)
    neg_notes = notes[~notes['HADM_ID'].isin(useful_hadms)].sample(frac=1, random_state=42)
    def get_valid_sentences(df_subset, label):
        valid = []
        # Keywords that suggest a sentence is valuable for ICD coding
        medical_keywords = ['history', 'diagnosis', 'acute', 'chronic', 'syndrome', 'pain', 'failure', 'patient']
        
        for text in df_subset['TEXT'].dropna():
            # Basic cleanup: remove newlines so sentence splitting is more accurate
            clean_text = str(text).replace('\n', ' ')
            sentences = sent_tokenize(clean_text)
            
            for s in sentences:
                words = s.split()
                # Use your 20-128 word limit
                if 20 <= len(words) <= 128:
                    s_lower = s.lower()
                    # For LABEL 1: Only keep it if it looks like a real clinical statement
                    if label == 1:
                        if any(kw in s_lower for kw in medical_keywords):
                            valid.append({'text': s.strip(), 'label': label})
                    
                    # For LABEL 0: Just keep it (Nursing/Social work text is naturally less codable)
                    else:
                        valid.append({'text': s.strip(), 'label': label})
        return valid

    print("Extracting meaningful sentences...")
    pos_data = get_valid_sentences(pos_notes, 1)
    neg_data = get_valid_sentences(neg_notes, 0)

    # --- 4. Balance the Labels ---
    # Find out which list is smaller and match the other one to it
    target_size = min(len(pos_data), len(neg_data))
    print(f"Balancing dataset to {target_size} rows per label...")
    
    final_data = pos_data[:target_size] + neg_data[:target_size]
    
    # Shuffle the final list so labels 0 and 1 are mixed
    train_df = pd.DataFrame(final_data).sample(frac=1).reset_index(drop=True)
    
    # Save the file
    output_file = os.path.join(path, 'balanced_sbert_train.csv')
    train_df.to_csv(output_file, index=False)
    
    print(f"SUCCESS!")
    print(f"Saved {len(train_df)} rows to: {output_file}")
    print(f"Label Counts:\n{train_df['label'].value_counts()}")

if __name__ == "__main__":
    prepare_balanced_data()