


import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Starting training...")

# LOAD DATA -----------------------------------------------------------------------------------------
file_name = "final_bert_dataset.csv"
df = pd.read_csv(file_name)  

# shuffle whole dataset 
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)


# PRE-PROCESSING -----------------------------------------------------------------------------------------
# def clean_text(text):
#     text = text.lower()
#     text = text.strip()
#     text = " ".join(text.split())
#     return text

import re
def clean_text(text):
    text = text.lower()
    # replace newlines/tabs with space
    text = re.sub(r"[\n\r\t]+", " ", text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()
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
# model_name = "emilyalsentzer/Bio_ClinicalBERT"
model_name = "/scratch/user/kiana.shen22/Bio_ClinicalBERT"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)


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
    num_labels=2,
    local_files_only=True
)

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=2
# )


# METRICS  -----------------------------------------------------------------------------------------
# def compute_metrics(pred):
#     logits, labels = pred
#     preds = logits.argmax(axis=1)

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         labels,
#         preds,
#         average="binary"
#     )

#     acc = accuracy_score(labels, preds)

#     return {
#         "accuracy": acc,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     }
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# NICE METRICS OUTPUT -----------------------------------------------------------------------------------------
class PrettyMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(
            f"Epoch {state.epoch:.2f} | "
            f"Loss: {metrics.get('eval_loss', 0):.4f} | "
            f"Acc: {metrics.get('eval_accuracy', 0):.4f} | "
            f"F1: {metrics.get('eval_f1', 0):.4f} | "
            f"Prec: {metrics.get('eval_precision', 0):.4f} | "
            f"Rec: {metrics.get('eval_recall', 0):.4f}"
        )

# TRAINING SETUP  -----------------------------------------------------------------------------------------

# training_args = TrainingArguments(
#     output_dir="./clinicalbert_results",

#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_strategy="epoch",   # prints training loss once per epoch

#     learning_rate=2e-5,                 #hypeparams
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=10,
#     weight_decay=0.01,

#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,

#     logging_dir="./logs",
#     report_to="none",           # prevents wandb/tensorboard issues
#     disable_tqdm=True          # turns off progress bar visible
# )
training_args = TrainingArguments(
    output_dir="/scratch/user/kiana.shen22/clinicalbert_results",

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",   # prints training loss once per epoch

    learning_rate=2e-5,                 #hypeparams
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    logging_dir="/scratch/user/kiana.shen22/logs",
    report_to="none",           # prevents wandb/tensorboard issues
    disable_tqdm=True          # turns off progress bar visible
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[PrettyMetricsCallback()],
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
# trainer.save_model("./clinicalbert_icd_classifier")
# tokenizer.save_pretrained("./clinicalbert_icd_classifier")

trainer.save_model("/scratch/user/kiana.shen22/clinicalbert_icd_classifier")
tokenizer.save_pretrained("/scratch/user/kiana.shen22/clinicalbert_icd_classifier")