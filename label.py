import pandas as pd
import nltk
import os
import re
import random
from nltk.tokenize import sent_tokenize

# --- 1. Initial Setup & NLTK Downloads ---
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except:
    pass
useful_patterns = [
    # Explicit diagnoses
    r'diagnosed with', r'dx of', r'admitted for', r'treatment for',
    r'history of', r'with.*cancer', r'with.*diabetes', r'with.*hypertension',
    
    # Specific conditions
    r'acute', r'chronic', r'secondary to', r'due to', r'related to',
    r'complicated by', r'with.*comorbidity', r'major'
]
symptom_patterns = [
    r'complains of', r'presented with', r'c/o', r'reports',
    r'endorses', r'experiencing', r'symptoms include',
    r'fever', r'pain', r'shortness of breath', r'nausea',
    r'vomiting', r'diarrhea', r'cough', r'fatigue'
]
procedure_patterns = [
    r'underwent', r'performed', r'procedure', r'surgery',
    r'biopsy', r'resection', r'repair', r'replacement',
    r'intubation', r'ventilation', r'dialysis'
]
medication_patterns = [
    r'prescribed', r'medication', r'dose', r'mg', r'mcg',
    r'administered', r'taking', r'on.*therapy'
]
lab_patterns = [
    r'lab.*show', r'results.*reveal', r'elevated', r'decreased',
    r'positive for', r'negative for', r'count of',
    r'blood pressure', r'heart rate', r'temperature'
]
not_useful_patterns = [
    # Administrative
    r'patient seen by', r'follow up', r'appointment',
    r'discussed with', r'explained to', r'patient instructed',
    
    # Negative/rule-out (unless specifying what was ruled out)
    r'no acute', r'no evidence of', r'negative for',
    r'denies', r'without', r'excluding', r'non', r'no', r'unremarkable'
    
    # Demographic
    r'year old', r'male|female', r'presenting for',
    
    # Generic/Non-specific
    r'patient is stable', r'will continue', r'to follow',
    r'as above', r'see below', r'refer to',

    r'admission date', r'discharge date', r'date of birth'
]
df = pd.read_csv('train_data_initial_with_diagnosis.csv')
def find_label(text, original_label):
    txt = str(text).lower()
    #count useful indicators
    useful_count = 0
    for pattern in useful_patterns+symptom_patterns+procedure_patterns:
        if re.search(pattern, txt):
            useful_count+=1
    for pattern in lab_patterns+medication_patterns:
        if re.search(pattern, txt):
            useful_count+=0.5
    not_use = 0
    for pattern in not_useful_patterns:
        if re.search(pattern, txt):
            not_use+=1
    print('useful count', useful_count)
    print('not useful', not_use)
    if useful_count>not_use:
        return 1
    elif not_use>useful_count:
        return 0
    else:
        return original_label

df = pd.read_csv('train_data_initial_with_diagnosis.csv')
df['label'] = df.apply(
        lambda row: find_label(row['text'], row['label']), 
        axis=1
    )
df.to_csv('train_data_initial_with_diagnosis.csv', index=False)

