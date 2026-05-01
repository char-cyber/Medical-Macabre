import pandas as pd
import re
from collections import Counter
df = pd.read_csv('train_data-text_and_labels.csv')
useful = df[df['label']==1]['text']
notuseful = df[df['label']==0]['text']
common = ['of', 'and', 'was', 'the', 'to', 'on', 'in', 'an', 'a','or','for', 'at', 'is', 'that', 'be', 'are', 'he', 'his', 'she', 'her']
common = set(common)
words = Counter()
for txt in useful:
    for w in txt.lower().split():
        if w not in common:
            words[w]+=1
nwords = Counter()
uwords = []
uwords.extend(words.most_common(20))
print(uwords)
nowords = []
for txt in notuseful:
    for w in txt.lower().split():
        if w not in common:
            nwords[w]+=1
nowords.extend(nwords.most_common(20))
print(nowords)

#yeswords: discharge, patient,history, blood, diagnosis, fracture
#nowords: evidence, no , impression,