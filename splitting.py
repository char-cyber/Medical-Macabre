import spacy
import pandas as pd
from label import *
from words import *
import re
try:
    nlp = spacy.load("en_core_sci_md")
except OSError:
    spacy.cli.download("en_core_sci_md")
    nlp = spacy.load("en_core_sci_md")
df = pd.read_csv('train_data-text_and_labels.csv')
yeses = df[df['label']==1]
nos = df[df['label']==0]
def get_valid_sentences(df_subset, label):
    valid = []
    for text in df_subset['text'].dropna():
        sections = split_sections(text)
        for si in sections:
            doc = nlp(si)
            sentences = [sent.text for sent in doc.sents]
            for s in sentences:
                new_s = str(s).replace('\n', ' ')
                new_s = re.sub(r'_+', '\n', new_s)
                new_s = new_s.strip()
                if len(s)>5:
                    valid.append({'text': new_s, 'label': label})
    return valid
# pos_data = get_valid_sentences(yeses,1)
# neg_data = get_valid_sentences(nos, 0)      
# final_data = pos_data + neg_data
# train_df = pd.DataFrame(final_data).reset_index(drop=True)
# train_df = train_df.drop_duplicates(subset=['text'], keep='first')
# output_file = 'train_first.csv'
# train_df.to_csv(output_file) #see if we need to set to true
# train_df.to_csv(output_file)
# train_df['label'] = train_df.apply(
#     lambda row: find_label(row['text'],row['label']), 
#     axis=1
# )
# train_df = train_df.drop_duplicates(subset=['text'], keep='first')
# train_df.to_csv('train_labeled.csv')

notes = pd.read_csv('NOTEEVENTS.csv')
diagnoses = pd.read_csv('DIAGNOSES_ICD.csv')
useful_hadms = diagnoses['HADM_ID'].unique()
# Take the top 20 notes for each category to extract sentences from
pos_notes = notes[notes['HADM_ID'].isin(useful_hadms)].sample(n=10, random_state=21)
pos_notes.columns= pos_notes.columns.str.lower()
neg_notes = notes[~notes['HADM_ID'].isin(useful_hadms)].sample(n=10, random_state=21)
neg_notes.columns = neg_notes.columns.str.lower()
#TEXT is in column TEXT
pos_data = get_valid_sentences(pos_notes, 1)
neg_data = get_valid_sentences(neg_notes, 0)
final_data = pos_data + neg_data
train_df = pd.DataFrame(final_data).reset_index(drop=True)
train_df = train_df.drop_duplicates(subset=['text'], keep='first')
output_file = 'data3.csv'
train_df.to_csv(output_file)
train_df['label'] = train_df.apply(
    lambda row: find_label(row['text'],row['label']), 
    axis=1
)
train_df = train_df.drop_duplicates(subset=['text'], keep='first')
train_df.to_csv('d3label.csv')


