import pandas as pd

import re

df = pd.read_csv('lot_of_data.csv')
# notes['text'] = notes['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
# notes.to_csv('d3label_cleaned.csv', index = False)

shuffled_df = df.sample(frac=1).reset_index(drop=True)
combined = shuffled_df.drop_duplicates(subset=['text'], keep='first')
combined.to_csv('data_test.csv')