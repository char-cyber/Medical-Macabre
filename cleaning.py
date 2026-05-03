import pandas as pd
import re

notes = pd.read_csv('labeled_data.csv')
notes['text'] = notes['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
notes.to_csv('labeled_data_cleaned.csv', index=False)