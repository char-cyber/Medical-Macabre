import pandas as pd

train_labeled = pd.read_csv('train_labeled_clean.csv')
labeled_data = pd.read_csv('labeled_data_cleaned.csv')
if 'row_id' in train_labeled.columns:
    train_labeled = train_labeled.drop(columns=['row_id'])
if 'row_id' in labeled_data.columns:
    labeled_data = labeled_data.drop(columns=['row_id'])
combined = pd.concat([train_labeled, labeled_data], ignore_index=True)
combined = combined.drop_duplicates(subset=['text'], keep='first')
combined.to_csv('combined_labeled_data.csv')