# FOR GRACE 
# import sys
# sys.path.insert(0, "/scratch/user/kiana.shen22/python_libs")

# regular code
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# USED TO SPLIT NOTES INTO SENTENCES!!!
# from words import split_sections
from splitting_bert import get_valid_sentences

# ------------- START MODEL PREDICTIONS -------------

# # load saved model
# model_path = "./clinicalbert_icd_classifier"
# # model_path = "/scratch/user/kiana.shen22/clinicalbert_icd_classifier"
# print("Starting predicitions")

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# # load dataset
# test1 = r'.\data\test01_text_only.csv'
# test2 = r'.\data\test02_text_only.csv'
# test3 = r'.\data\test03_text_only.csv'

# test_list = [(test1, "test01"), (test2, "test02"), (test3, "test03")]

# # loop through all testing files
# for test, testNum in test_list:
#     # create df and read file
#     df = pd.read_csv(test)  
#     print("Read File:")
#     texts = df["text"].astype(str).tolist()

#     # list for predictions and probs
#     predictions = []
#     probabilities = []

#     batch_size = 16
#     threshold = 0.35

    # for note in texts:
    #     temp_df = pd.DataFrame({"text": [note]})
    #     sentence_data = get_valid_sentences(temp_df, label=0)

    #     sentences = [item["text"] for item in sentence_data]

    #     sentence_probs = []

    #     with torch.no_grad():
    #         for i in range(0, len(sentences), batch_size):
    #             batch_texts = sentences[i:i + batch_size]

    #             inputs = tokenizer(
    #                 batch_texts,
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=128,
    #                 return_tensors="pt"
    #             )

    #             inputs = {key: value.to(device) for key, value in inputs.items()}

    #             outputs = model(**inputs)
    #             probs = torch.softmax(outputs.logits, dim=1)

    #             sentence_probs.extend(probs[:, 1].cpu().tolist())

    #     # note-level heuristic
    #     if len(sentence_probs) == 0:
    #         note_prob = 0
    #     else:
    #         note_prob = max(sentence_probs)

    #     note_pred = int(note_prob >= threshold)

    #     predictions.append(note_pred)
    #     probabilities.append(note_prob)


    # output_df = pd.DataFrame({
    #     "row_id": df["row_id"],
    #     "prediction": predictions,
    #     "confidence": probabilities
    # })

    # saved_csv = testNum + "-pred.csv"
    # output_df.to_csv(saved_csv, index=False)
    # print("Saved predictions to", saved_csv)


# DEBUGGING ---------------------------------------------------------------------------------


# load saved model
model_path = "./clinicalbert_icd_classifier"
# model_path = "/scratch/user/kiana.shen22/clinicalbert_icd_classifier"
print("Starting predicitions")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# load dataset
test1 = r'.\data\test01_text_only.csv'
test2 = r'.\data\test02_text_only.csv'
test3 = r'.\data\test03_text_only.csv'

test_list = [(test1, "test01"), (test2, "test02"), (test3, "test03")]

# loop through all testing files
for test, testNum in test_list:
    # create df and read file
    df = pd.read_csv(test)  
    print("Read File:")
    texts = df["text"].astype(str).tolist()

    # list for predictions and probs
    predictions = []
    probabilities = []

    batch_size = 16
    threshold = 0.35

    # loop through all of the notes
    for note in texts:
        # print("NOTE -------------------------------------------------------")
        # print(note)
        # print("\n")

        sentences = get_valid_sentences(note)
        # print("Sentence List Length: ", len(sentences), "\n")
        sentence_probs = []

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_texts = sentences[i:i + batch_size]

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                inputs = {key: value.to(device) for key, value in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=1)
                # print("Sentence ", i, )
                # print("Sentence ", i, " Probability: ", probs)
                # print("\n")
                sentence_probs.extend(probs[:, 1].cpu().tolist())
        # note-level heuristic
        if len(sentence_probs) == 0:
            note_prob = 0
        else:
            note_prob = max(sentence_probs)

        note_pred = int(note_prob >= threshold)
        # print("NOTE LABEL: ", note_pred)
        # print("\n")
        # print("-------------------------------------------------------------------------------")

        predictions.append(note_pred)
        probabilities.append(note_prob)

    output_df = pd.DataFrame({
        "row_id": df["row_id"],
        "prediction": predictions,
        "confidence": probabilities
    })

    saved_csv = testNum + "-pred.csv"
    output_df.to_csv(saved_csv, index=False)
    print("Saved predictions to", saved_csv)