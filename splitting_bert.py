import spacy
import pandas as pd
import re
# try:
#     nlp = spacy.load("en_core_sci_md")
# except OSError:
#     spacy.cli.download("en_core_sci_md")
#     nlp = spacy.load("en_core_sci_md")
nlp = spacy.load("en_core_sci_md")
df = pd.read_csv('train_data-text_and_labels.csv')
yeses = df[df['label']==1]
nos = df[df['label']==0]
# def get_valid_sentences(df_subset, label):
#     valid = []
#     for text in df_subset['text'].dropna():
#         sections = split_sections(text)
#         for si in sections:
#             doc = nlp(si)
#             sentences = [sent.text for sent in doc.sents]
#             for s in sentences:
#                 new_s = str(s).replace('\n', ' ')
#                 new_s = re.sub(r'_+', '\n', new_s)
#                 new_s = new_s.strip()
#                 if len(s)>5:
#                     valid.append({'text': new_s, 'label': label})
#     return valid

# def split_sections(text):
#     #split into colons, blank lines, bulletpoints
#     if pd.isna(text):
#         return []
#     results = []
#     #split by blank lines 
#     sentences = re.split(r'\n\s*\n+', text)
#     #also split by sentences
#     for section in sentences:
#         if not section.strip():
#             continue
#         sct = re.split(r'(?=discharge diagnosis:|discharge condition:)', section, flags=re.IGNORECASE)
#         for s in sct:
#             s.replace("-", "")
#         results.extend(sct)
#     return results


def get_valid_sentences(text, min_length=20): #should i set min length higher?

    if not text or not isinstance(text, str):
        return []

    # Split by blank lines
    pre_split = re.split(r'\n\s*\n+', text)
    
    sections = []
    for section in pre_split:
        # Split by specific medical headers (keeps the header with the content)
        header_split = re.split(r'(?=discharge diagnosis:|discharge condition:)', section, flags=re.IGNORECASE)
        for s in header_split:
            cleaned_s = s.replace("-", "").strip()
            if cleaned_s:
                sections.append(cleaned_s)

    # NLP Sentence Segmentation
    final_sentences = []
    for section_text in sections:
        doc = nlp(section_text)
        
        for sent in doc.sents:
            # Cleanup & Formatting
            s_text = sent.text.replace('\n', ' ')      # Remove internal newlines
            s_text = re.sub(r'_+', '\n', s_text)       # Convert underscores to newlines
            s_text = s_text.strip()
            
            # Validation
            if len(s_text) > min_length:
                final_sentences.append(s_text)
                
    return final_sentences