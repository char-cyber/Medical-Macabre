
import pandas as pd
import re

#read the orignal data
original = pd.read_csv("train_data-text_and_labels.csv")
#cleaned dataset 
clean = pd.read_csv("train_labeled_clean.csv")
#Load Data 
original["text"] = original["text"].astype(str)
clean["text"] = clean["text"].astype(str)

#ensure text is string, remove extra spaces, remove leading spaces 
def normalize(s):
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()
#normalize full note text 
original["note_text_norm"] = original["text"].apply(normalize)
clean["sent_norm"] = clean["text"].apply(normalize)

matched_rows = [] #store matched results 
# loop through each sentence 
for _, sent_row in clean.iterrows():
    sent = sent_row["sent_norm"]
    #find notes that contain the sentence 
    match = original[original["note_text_norm"].str.contains(re.escape(sent), na=False)]
    
    if len(match) > 0:
        note = match.iloc[0]
        #store mapping info 
        matched_rows.append({
            "sentence_text": sent_row["text"],
            "sentence_label": sent_row["label"],
            "note_row_id": note["row_id"],
            "true_note_label": note["label"]
        })
#convert matched results to DataFrame
matched = pd.DataFrame(matched_rows)

note_stats = matched.groupby("note_row_id").agg(
    total_sentences=("sentence_text", "count"),
    useful_sentences=("sentence_label", "sum"),
    true_note_label=("true_note_label", "first")
).reset_index()
# compute how useful sentences exist in a sentence 
note_stats["percent_useful"] = note_stats["useful_sentences"] / note_stats["total_sentences"]
# test diff thresholds 
for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
    note_stats["predicted"] = (note_stats["percent_useful"] >= t).astype(int)
    acc = (note_stats["predicted"] == note_stats["true_note_label"]).mean()
    print(f"Threshold {t}: accuracy = {acc:.3f}")
#best threshold found 
threshold = 0.40
note_stats["predicted_label"] = (note_stats["percent_useful"] >= threshold).astype(int)
# find outliers 
outliers = note_stats[note_stats["predicted_label"] != note_stats["true_note_label"]]
#saved results 
matched.to_csv("matched_sentences_to_notes.csv", index=False)
note_stats.to_csv("note_level_threshold_analysis.csv", index=False)
outliers.to_csv("outlier_notes.csv", index=False)
#print summary 
print("Matched sentences:", len(matched), "out of", len(clean))
print("Notes matched:", len(note_stats))
print("Outliers:", len(outliers))