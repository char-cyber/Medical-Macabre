import pandas as pd

import re

# notes = pd.read_csv('d3label.csv')
# notes['text'] = notes['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
# notes.to_csv('d3label_cleaned.csv', index = False)

train_labeled = pd.read_csv('combined_labeled_data.csv')
labeled_data = pd.read_csv('d3label_cleaned.csv')
if 'row_id' in train_labeled.columns:
    train_labeled = train_labeled.drop(columns=['row_id'])
if 'row_id' in labeled_data.columns:
    labeled_data = labeled_data.drop(columns=['row_id'])
combined = pd.concat([train_labeled, labeled_data], ignore_index=True)
combined = combined.drop_duplicates(subset=['text'], keep='first')
combined.to_csv('combined_labeled_data1.csv')