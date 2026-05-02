

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Load saved model
# -----------------------------
model_path = "./clinicalbert_icd_classifier"
print("Starting predicitions")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -----------------------------
# Load dataset to test
# -----------------------------
test1 = "test01_text_only.csv"
test2 = "test02_text_only.csv"
test3= "test03_text_only.csv"

test_list = [(test1, "test01"), (test2, "test02"), (test3, "test03")]

test = "test03_text_only.csv"
testNum = "test3"

df = pd.read_csv(test)  # change this filename
# Change "text" if your column has a different name
print(df)
texts = df["text"].astype(str).tolist()

# -----------------------------
# Run predictions
# -----------------------------
predictions = []
probabilities = []

batch_size = 16

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

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
        preds = torch.argmax(probs, dim=1)

        predictions.extend(preds.cpu().tolist())
        probabilities.extend(probs[:, 1].cpu().tolist())

    # -----------------------------
    # Save output CSV
    # -----------------------------
    # df["label"] = predictions

    # saved_csv = testNum + "-pred.csv"
    # df.to_csv(saved_csv, index=False)
    output_df = pd.DataFrame({
        "row_id": df["row_id"],   # make sure this column exists
        "prediction": predictions
    })

    saved_csv = testNum + "-pred.csv"
    output_df.to_csv(saved_csv, index=False)
    print("Saved predictions to ", saved_csv)

# for test, testNum in test_list:

#     df = pd.read_csv(test)  # change this filename

#     # Change "text" if your column has a different name
#     texts = df["text"].astype(str).tolist()

#     # -----------------------------
#     # Run predictions
#     # -----------------------------
#     predictions = []
#     probabilities = []

#     batch_size = 16

#     with torch.no_grad():
#         for i in range(0, len(texts), batch_size):
#             batch_texts = texts[i:i + batch_size]

#             inputs = tokenizer(
#                 batch_texts,
#                 padding=True,
#                 truncation=True,
#                 max_length=128,
#                 return_tensors="pt"
#             )

#             inputs = {key: value.to(device) for key, value in inputs.items()}

#             outputs = model(**inputs)
#             logits = outputs.logits

#             probs = torch.softmax(logits, dim=1)
#             preds = torch.argmax(probs, dim=1)

#             predictions.extend(preds.cpu().tolist())
#             probabilities.extend(probs[:, 1].cpu().tolist())

#     # -----------------------------
#     # Save output CSV
#     # -----------------------------
#     # df["label"] = predictions

#     # saved_csv = testNum + "-pred.csv"
#     # df.to_csv(saved_csv, index=False)
#     output_df = pd.DataFrame({
#         "row_id": df["row_id"],   # make sure this column exists
#         "prediction": predictions
#     })

#     saved_csv = testNum + "-pred.csv"
#     output_df.to_csv(saved_csv, index=False)
#     print("Saved predictions to ", saved_csv)



# NOTE --data into sentences FIRST! then figure out note: final note score would be 1 (confidence level of model) --> score probabilty off of this