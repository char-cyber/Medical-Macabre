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

# df = pd.read_csv('train_data-text_and_labels.csv')
# useful = df[df['label']==1]['text']
# not_useful = df[df['label']==0]['text']

# vectorizer = CountVectorizer(ngram_range=(2,3))
# ngrams_matrix = vectorizer.fit_transform(useful)
# counts = ngrams_matrix.toarray().sum(axis=0)
# vc = vectorizer.get_feature_names_out()
# ngram_freq = sorted(zip(counts, vc), reverse=True)

# useful_words = []
# for count, ngram in ngram_freq[:5]:
#     useful_words.append(ngram)
#     print(f"{ngram}: {count}")

# yeses = df[df['label']==1]
# nos = df[df['label']==0]

placeholders = {}
protected_pattern = r'\[\*\*[^\]]*\*\*\]'
protected_hyphens = r'\b\w+-\w+\b'

def replace_protected(match):
    placeholder = f"__PROTECTED_{len(placeholders)}__"
    placeholders[placeholder] = match.group(0)
    return placeholder

def split_sections(text):
    if pd.isna(text):
        return []
    results = []
    #split by blank lines 
    sentences = re.split(r'\n\s*\n+', text)
    for section in sentences:
        if not section.strip():
            continue
        sct = re.split(r'(?=discharge diagnosis:|discharge condition:)', section, flags=re.IGNORECASE)
        for s in sct:
            s = s.replace("-", "")
        results.extend(sct)
    return results

def split_by_colons(text):
    if not text or len(text.strip()) < 2:
        return []
    
    text = text.strip()
    if text.count(":") == 0:
        return [text] if len(text) > 2 else []
    if text.strip().endswith(":"):
        return [text] 
    results = []
    parts = [p.strip() for p in text.split(":")]
    if len(parts) == 2:
        left, right = parts[0], parts[1]
        if left and right and len(left) > 0 and len(right) > 1:
            if not right.endswith(':') and len(right) > 1:
                results.append(f"{left}: {right}")
        return results
    i = 0
    while i < len(parts) - 1:
        current = parts[i]
        next_part = parts[i + 1]
        if not current or not next_part:
            i += 1
            continue
        if len(current) < 50 and next_part and not next_part.endswith(':'):
            paired = f"{current}: {next_part}"
            if len(paired) > 5 and len(current) > 1 and len(next_part) > 1:
                results.append(paired)
            i += 2
        else:
            if len(current) > 3:
                results.append(current)
            i += 1
    if i == len(parts) - 1 and len(parts[-1]) > 3:
        if not parts[-1].endswith(':'):
            results.append(parts[-1])
    seen = set()
    unique_results = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique_results.append(r)
    
    return unique_results if unique_results else ([text] if len(text) > 3 else [])

def get_valid_sentences(df_subset, label):
    valid = []
    for idx, text in enumerate(df_subset['text'].dropna()):
        sections = split_sections(text)
        for si in sections:
            sic = re.sub(r'_+', '\n', si)
            if 'discharge diagnosis' in sic.lower() or 'discharge condition' in sic.lower() or 'secondary diagnosis' in sic.lower():
                content = sic
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if len(line) > 5:
                        colon_splits = split_by_colons(line)
                        for cs in colon_splits:
                            if len(cs) > 5:
                                valid.append({'text': cs, 'label': label})
                continue
            clean_text = str(sic).replace('\n', ' ')
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            if not clean_text or len(clean_text) < 5:
                continue
            sentences = sent_tokenize(clean_text)
            for s in sentences:
                s = s.strip()
                if len(s) < 5:
                    continue
                hyphen = False
                inter = []
                if '-' in s and not re.search(r'\d+-\d+', s):
                    hyphen = True
                    placeholders.clear()
                    tmp_text = re.sub(protected_pattern, replace_protected, s)
                    tmp_text = re.sub(protected_hyphens, replace_protected, tmp_text)
                    pts = re.split(r'-\s+', tmp_text)
                    for p in pts:
                        p = p.strip()
                        if p:
                            for place, original in placeholders.items():
                                if place in p:
                                    p = p.replace(place, original)
                            if len(p) > 5:
                                inter.append(p)
                if not hyphen:
                    inter.append(s)
                for pi in inter:
                    if ':' in pi:
                        colon_parts = split_by_colons(pi)
                        for v in colon_parts:
                            v = v.strip()
                            if len(v) > 5:
                                valid.append({'text': v, 'label': label})
                    else:
                        pi_clean = pi.strip()
                        if len(pi_clean) > 5:
                            valid.append({'text': pi_clean, 'label': label})
    
    return valid

# pos_data = get_valid_sentences(yeses, 1)
# neg_data = get_valid_sentences(nos, 0)

# final_data = pos_data + neg_data
# train_df = pd.DataFrame(final_data).reset_index(drop=True)
# train_df = train_df.drop_duplicates(subset=['text'], keep='first')
# output_file = 'train_data_initial_with_diagnosis.csv'
# train_df.to_csv(output_file, index=False)
