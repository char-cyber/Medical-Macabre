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
section_headers = ['Discharge Condition', 'Discharge Instructions', 'Social History', 'Family History', 'Chief Complaint', 'Past Medical History', 'Physical Exam', 'Medications', 'Allergies']
def remove_section_headers(txt):
    for h in section_headers:
        pattern = re.compile(r'\b' + re.escape(h) + r'\s*:?\s*\n?', re.IGNORECASE)
        txt = pattern.sub('', txt)
    return txt.strip()

def is_likely_header(text):
    text_stripped = text.strip()
    if not text_stripped:
        return True
    common_header_words = ['history', 'diagnosis', 'condition', 'exam', 'medication', 'allergy', 'complaint', 'instruction', 'summary', 'assessment']
    if len(text_stripped) < 30 and text_stripped[0].isupper():
        if any(word in text_stripped.lower() for word in common_header_words):
            return True
    return False
    
def split_by_colons(text):
    if text.count(":")==0:
        return [text.strip()]
    #check if text ends with colon (its a header, not pair)
    if text.strip().endswith(":"):
        return [text]
    results = []
    pts = [p.strip() for p in text.split(":")]
    i = 1
    while i< len(pts)-1:
        first = pts[i]
        second = pts[i+1]
        if not second.endswith(":"):
            paired = f"{first} : {second}"
            results.append(paired)
        else:
            #add both individually
            results.append(first)
            results.append(second)
        i+=2
    #what about last value? #add individually 
    if results:
        return results
    else:
        return text
        
def get_valid_sentences(df_subset, label):
    valid = []
    for text in df_subset['text'].dropna():
        sections = split_sections(text)
        for si in sections: 
            #clean_text = str(si).replace('\n', ' ')
            sic = re.sub(r'_+', '\n', si)
            if 'discharge diagnosis' in sic.lower() or 'discharge condition' in sic.lower() or 'secondary diagnosis' in sic.lower():
                #content = re.sub(r'discharge diagnosis\s*:?\s*', '', sic, flags=re.IGNORECASE)
                content = re.sub(r'discharge condition\s*:?\s*', '', sic, flags=re.IGNORECASE)
                content = re.sub(r'secondary diagnosis\s*:?\s*', '', content, flags=re.IGNORECASE)
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    valid.append({'text': line, 'label': label})
                continue
            clean_text = str(sic).replace('\n', ' ')
            val = remove_section_headers(clean_text)
            sentences = sent_tokenize(val) #try fallback to simple splitting
            #split up discharge sections anyw
            for s in sentences:
                #placeholders = {}
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
                    values = split_by_colons(pi)
                    for v in values:
                        valid.append({'text': v, 'label': label})
    return valid
pos_data = get_valid_sentences(yeses, 1)
neg_data = get_valid_sentences(nos, 0)
final_data = pos_data + neg_data
train_df = pd.DataFrame(final_data).reset_index(drop=True)
train_df = train_df.drop_duplicates(subset=['text'], keep='first')
output_file = 'train_data_initial_with_diagnosis.csv'
# train_df.to_csv(output_file) #see if we need to set to true
if __name__ == "__main__":
    train_df.to_csv(output_file)
