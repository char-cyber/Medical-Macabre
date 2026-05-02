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
from words import *
useful_patterns = [
    # Explicit diagnoses
    r'diagnosed with', r'dx of', r'admitted for', r'treatment for',
    r'history of', r'with.*cancer', r'with.*diabetes', r'with.*hypertension',
    
    # Specific conditions
    r'acute', r'chronic', r'secondary to', r'due to', r'related to',
    r'complicated by', r'with.*comorbidity', r'major', r'significant', r'hemorrhage', r'left', r'right'
]
symptom_patterns = [
    r'complains of', r'presented with', r'c/o', r'reports',
    r'endorses', r'experiencing', r'symptoms include',
    r'fever', r'pain', r'shortness of breath', r'nausea',
    r'vomiting', r'diarrhea', r'cough', r'fatigue', r'bleeding'
]
procedure_patterns = [
    r'underwent', r'performed', r'procedure', r'surgery',
    r'biopsy', r'resection', r'repair', r'replacement',
    r'intubation', r'ventilation', r'dialysis'
]
condition_keywords = [
        r'diabetes', r'hypertension', r'cancer', r'tumor', r'mass',
        r'infection', r'pneumonia', r'sepsis', r'failure', r'disease',
        r'syndrome', r'disorder', r'injury', r'fracture', r'copd',
        r'asthma', r'cellulitis', r'anemia', r'kidney', r'liver',r'intestin', r'fluid'
        r'thyroid', r'depression', r'anxiety', r'stroke',
        r'mi', r'myocardial infarction', r'cva', r'tia',r'blood', r'throat',r'abdomen', r'quadrant',r'abdominal', r'rectal'
    ]
medication_patterns = [
    r'administered'
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
    r'no acute', r'no evidence', r'negative for', r'maintain', r'remain'
    r'denies', r'without', r'excluding', r'non', r'no', r'unremarkable', r'unclear', r'unknown', r'unchanged'
    
    # Demographic
    r'year old', r'male|female', r'presenting for', r'lives', r'resides', r'home'
    
    # Generic/Non-specific
    r'patient is stable', r'will continue', r'to follow',
    r'as above', r'see below', r'refer to',

    r'admission date', r'discharge date', r'date of birth'
]
#df = pd.read_csv('train_data_initial_with_diagnosis.csv')
def find_label(text, original_label):
    txt = str(text).lower()
    #count useful indicators
    useful_count = 0
    for pattern in useful_patterns+symptom_patterns+procedure_patterns+condition_keywords+lab_patterns+medication_patterns:
        if re.search(pattern, txt):
            useful_count+=1
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
notes = pd.read_csv('NOTEEVENTS.csv')
diagnoses = pd.read_csv('DIAGNOSES_ICD.csv')
useful_hadms = diagnoses['HADM_ID'].unique()
# Take the top 20 notes for each category to extract sentences from
pos_notes = notes[notes['HADM_ID'].isin(useful_hadms)].sample(n=10, random_state=42)
pos_notes.columns= pos_notes.columns.str.lower()
neg_notes = notes[~notes['HADM_ID'].isin(useful_hadms)].sample(n=10, random_state=42)
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

