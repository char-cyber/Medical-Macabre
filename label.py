import pandas as pd
import nltk
import os
import re
import random
from nltk.tokenize import sent_tokenize
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except:
    pass
from wok import *
DIAGNOSIS_PATTERNS = [
    r'\bdiagnosed with\b',
    r'\bdx of\b',
    r'\bimpression of\b',
    r'\bconsistent with\b',
    r'\bconfirmed\b',
]
CONDITION_KEYWORDS = [
    r'diabetes', r'hypertension', r'cancer', r'tumor', r'mass',
    r'infection', r'pneumonia', r'sepsis', r'heart failure',
    r'kidney failure', r'liver failure', r'syndrome', r'disorder',
    r'injury', r'fracture', r'copd', r'asthma', r'anemia',
    r'stroke', r'myocardial infarction', r'cva', r'tia',
    r'depression', r'anxiety', r'thyroid'
]
SYMPTOM_PATTERNS = [
    r'\bcomplains of\b',
    r'\bpresented with\b',
    r'\breports\b',
    r'\bc/o\b',
    r'\bpain\b',
    r'\bfever\b',
    r'\bcough\b',
    r'\bnausea\b',
    r'\bshortness of breath\b',
    r'\bfatigue\b'
]
PROCEDURE_PATTERNS = [
    r'\bunderwent\b',
    r'\bperformed\b',
    r'\bsurgery\b',
    r'\bbiopsy\b',
    r'\bresection\b',
    r'\bdialysis\b',
    r'\bintubation\b',
    r'\bventilation\b'
]
LAB_PATTERNS = [
    r'\belevated\b',
    r'\bdecreased\b',
    r'\bpositive\b',
    r'\bnegative\b',
    r'\bwhite blood cell\b',
    r'\bhemoglobin\b',
    r'\bplatelet\b',
    r'\bblood pressure\b',
    r'\bheart rate\b'
]
NON_RELEVANT_PATTERNS = [
    r'\bfollow up\b',
    r'\bappointment\b',
    r'\bpatient instructed\b',
    r'\bdiscussed with\b',
    r'\bwill continue\b',
    r'\bstable\b',
    r'\bdischarge\b',
    r'\badmission date\b',
    r'\bdate of birth\b',
    r'\bmale\b',
    r'\bfemale\b',
    r'\blives with\b',
    r'\bresides\b',
    r'admission date', r'discharge date', r'date of birth', r'no evidence', r'unremarkable', r'unclear', r'unknown', r'unchanged',r'not consistent'
]
def find_label(text, original_label):
    txt = str(text).lower()

    score = 0

    # strong positive
    for p in DIAGNOSIS_PATTERNS:
        if re.search(p, txt):
            score += 5

    # conditions
    for p in CONDITION_KEYWORDS:
        if re.search(p, txt):
            score += 3

    # procedures
    for p in PROCEDURE_PATTERNS:
        if re.search(p, txt):
            score += 4

    # labs
    for p in LAB_PATTERNS:
        if re.search(p, txt):
            score += 2

    # symptoms
    for p in SYMPTOM_PATTERNS:
        if re.search(p, txt):
            score += 1

    # negative signals
    for p in NON_RELEVANT_PATTERNS:
        if re.search(p, txt):
            score -= 3

    # final decision rule
    if score >= 3:
        return 1
    elif score <= -2:
        return 0
    else:
        return original_label
#df = pd.read_csv('train_data_initial_with_diagnosis.csv')

notes = pd.read_csv('NOTEEVENTS.csv')
diagnoses = pd.read_csv('DIAGNOSES_ICD.csv')
useful_hadms = diagnoses['HADM_ID'].unique()
# Take the top 20 notes for each category to extract sentences from
pos_notes = notes[notes['HADM_ID'].isin(useful_hadms)].sample(n=100, random_state=42)
pos_notes.columns= pos_notes.columns.str.lower()
neg_notes = notes[~notes['HADM_ID'].isin(useful_hadms)].sample(n=100, random_state=42)
neg_notes.columns = neg_notes.columns.str.lower()
#TEXT is in column TEXT
pos_data = get_valid_sentences(pos_notes, 1)
neg_data = get_valid_sentences(neg_notes, 0)
final_data = pos_data + neg_data
train_df = pd.DataFrame(final_data).reset_index(drop=True)
train_df = train_df.drop_duplicates(subset=['text'], keep='first')
output_file = 'additional_data.csv'
train_df.to_csv(output_file)
train_df['label'] = train_df.apply(
    lambda row: find_label(row['text'],row['label']), 
    axis=1
)
train_df = train_df.drop_duplicates(subset=['text'], keep='first')
train_df.to_csv('labeled_add_data.csv')

