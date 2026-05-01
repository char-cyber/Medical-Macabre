


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd 
# from sklearn.model_selection import train_test_split
# import nltk
# from nltk.tokenize import word_tokenize
# import string
# import re
# from tqdm import tqdm
# import math
# from datasets import Dataset
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer
# )

import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# LOAD DATA -----------------------------------------------------------------------------------------
df = pd.read_csv("balanced_sbert_train.csv")
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)


# PRE-PROCESSING -----------------------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = " ".join(text.split())
    return text

df["text"] = df["text"].apply(clean_text)

# TRAIN/val/test_split -----------------------------------------------------------------------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

# LOAD CLINICAL BERT  -----------------------------------------------------------------------------------------
model_name = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Apply tokenization
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# LOAD CLINICAL BERT CLASSIFIER  -----------------------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# METRICS  -----------------------------------------------------------------------------------------
def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# # TRAINING SETUP  -----------------------------------------------------------------------------------------
# training_args = TrainingArguments(
#     output_dir="./clinicalbert_results",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     logging_dir="./logs",
#     logging_steps=50
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     processing_class=tokenizer,
#     compute_metrics=compute_metrics,
# )

# # TRAIN AND TEST  -----------------------------------------------------------------------------------------
# trainer.train()

# results = trainer.evaluate(test_ds)
# print(results)

# # SAVE MODEL  -----------------------------------------------------------------------------------------
# trainer.save_model("./clinicalbert_icd_classifier")
# tokenizer.save_pretrained("./clinicalbert_icd_classifier")



# TRAINING SETUP  -----------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./clinicalbert_results",

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",   # prints training loss once per epoch

    learning_rate=2e-5,                 #hypeparams
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    logging_dir="./logs",
    report_to="none",           # prevents wandb/tensorboard issues
    disable_tqdm=False          # keeps progress bar visible
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# TRAIN AND TEST  -----------------------------------------------------------------------------------------
train_result = trainer.train()

print("\n==================== EPOCH LOGS ====================")
for log in trainer.state.log_history:
    if "epoch" in log:
        print(log)

print("\n==================== FINAL TEST RESULTS ====================")
results = trainer.evaluate(test_ds)
print(results)

# SAVE MODEL  -----------------------------------------------------------------------------------------
trainer.save_model("./clinicalbert_icd_classifier")
tokenizer.save_pretrained("./clinicalbert_icd_classifier")