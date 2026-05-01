import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import os
import random
from nltk.tokenize import sent_tokenize
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except:
    pass
df = pd.read_csv('train_data-text_and_labels.csv')
useful = df[df['label']==1]['text']
not_useful = df[df['label']==0]['text']
vectorizer = CountVectorizer(ngram_range=(2,3))
ngrams_matrix = vectorizer.fit_transform(useful)
counts = ngrams_matrix.toarray().sum(axis=0)
vc = vectorizer.get_feature_names_out()
ngram_freq = sorted(zip(counts, vc), reverse=True)
useful_words = []
for count, ngram in ngram_freq[:5]:
    useful_words.append(ngram)
    print(f"{ngram}: {count}")
yeses = df[df['label']==1]
nos = df[df['label']==0]
print(yeses)
placeholders = {}
protected_pattern = r'\[\*\*[^\]]*\*\*\]'
protected_hyphens = r'\b\w+-\w+\b'
def replace_protected(match):
    placeholder = f"__PROTECTED_{len(placeholders)}__"
    placeholders[placeholder] = match.group(0)
    return placeholder

def split_spaces(text):
    parts = re.split(r'\s{3,}|\t+', text)
    return [p.strip() for p in parts if p.strip()]
def split_sections(text):
    #split into colons, blank lines, bulletpoints
    if pd.isna(text):
        return []
    results = []
    #split by blank lines 
    sentences = re.split(r'\n\s*\n+', text)
    #also split by sentences
    for section in sentences:
        if not section.strip():
            continue
        sct = re.split(r'(?=discharge diagnosis:|discharge condition:)', section, flags=re.IGNORECASE)
        for s in sct:
            s.replace("-", "")
        results.extend(sct)
    return results

        #split colons
section_headers = ["Discharge Diagnosis", 'Discharge Condition', 'Discharge Instructions', 'Social History', 'Family History', 'Chief Complaint', 'Past Medical History', 'Physical Exam', 'Medications', 'Allergies']
def remove_section_headers(txt):
    for h in section_headers:
        pattern = re.compile(r'\b' + re.escape(h) + r'\s*:?\s*\n?', re.IGNORECASE)
        txt = pattern.sub('', txt)
    return txt.strip()
def get_valid_sentences(df_subset, label):
    valid = []
    for text in df_subset['text'].dropna():
        sections = split_sections(text)
        for si in sections: 
            #clean_text = str(si).replace('\n', ' ')
            sic = re.sub(r'_+', '', si)
            if 'discharge diagnosis' in sic.lower() or 'secondary diagnosis' in sic.lower():
                content = re.sub(r'discharge diagnosis\s*:?\s*', '', sic, flags=re.IGNORECASE)
                content = re.sub(r'discharge condition\s*:?\s*', '', content, flags=re.IGNORECASE)
                content = re.sub(r'secondary diagnosis\s*:?\s*', '', content, flags=re.IGNORECASE)
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    valid.append({'text': line, 'label': label})
                continue
            clean_text = str(sic).replace('\n', ' ')
            val = remove_section_headers(clean_text)
            sentences = sent_tokenize(val)
            #split up discharge sections anyw
            for s in sentences:
                hyphen = False
                inter = []
                if '-' in s:
                    hyphen = True
                    tmp_text = re.sub(protected_pattern, replace_protected, s)
                    tmp_text = re.sub(protected_hyphens, replace_protected, tmp_text)
                    pts = re.split(r'-\s+',tmp_text) #split on dash followed by space
                    for p in pts:
                        p = p.strip()
                        if p:
                            for place, original in placeholders.items():
                                if place in p:
                                    p = p.replace(place, original)
                            if len(p)>5:
                                inter.append(p)
                #now check for colons and split those up
                if not hyphen:
                    inter.append(s) #get the whole sentence
                for pi in inter:
                    if pi.count(":")>1:
                        pts = [p.strip() for p in pi.split(":")]
                        pts = [word for word in pts if word not in section_headers]
                        for i in range(1, len(pts),2):
                            arr = pts[i-1]+" : "+pts[i]
                            arr = re.sub(r'\s+', ' ', arr)
                            valid.append({'text': arr, 'label': label})
                    else:
                        if len(pi)>5:
                            pi = re.sub(r'\s+', ' ', pi.strip())
                            valid.append({'text': pi, 'label': label})
    return valid
pos_data = get_valid_sentences(yeses, 1)
neg_data = get_valid_sentences(nos, 0)
final_data = pos_data + neg_data
train_df = pd.DataFrame(final_data).reset_index(drop=True)
output_file = 'train_data_initial.csv'
train_df.to_csv(output_file) #see if we need to set to true