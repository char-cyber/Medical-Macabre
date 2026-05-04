import torch
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Running BASELINE inference (no fine-tuning)...")

# LOAD DATA -----------------------------------------------------------------------------------------
file_name = "train_labeled_clean.csv"
df = pd.read_csv(file_name)

# Shuffle whole dataset (same seed as training script for consistency)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

print(f"\nLabel distribution:\n{df['label'].value_counts()}")

# PRE-PROCESSING -----------------------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# SAME TRAIN/VAL/TEST SPLIT AS TRAINING SCRIPT ------------------------------------------------------------
# Using the same random_state=42 so the test set is identical to what your fine-tuned model will see
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

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"\nRunning baseline inference on TEST SET only ({len(test_df)} samples)...")

test_ds = Dataset.from_pandas(test_df)

# LOAD CLINICAL BERT -----------------------------------------------------------------------------------------
model_name = "/scratch/user/charu7465/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

test_ds = test_ds.map(tokenize, batched=True)
test_ds = test_ds.rename_column("label", "labels")
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# LOAD VANILLA (PRE-TRAINED, NOT FINE-TUNED) CLINICALBERT -------------------------------------------------
# This loads the classification head randomly initialized — this IS the baseline
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    local_files_only=True
)

print("\nNOTE: Classification head is randomly initialized — this is the true pre-trained baseline.")

# METRICS -----------------------------------------------------------------------------------------
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

# USE TRAINER JUST FOR INFERENCE (no training) -----------------------------------------------------------
# A minimal TrainingArguments just to satisfy Trainer — no actual training happens
training_args = TrainingArguments(
    output_dir="/scratch/user/charu7465/baseline_results",
    per_device_eval_batch_size=16,
    report_to="none",
    disable_tqdm=True,
    no_cuda=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# RUN BASELINE INFERENCE -----------------------------------------------------------------------------------------
baseline_results = trainer.evaluate()

print("\n==================== BASELINE RESULTS (Pre-Fine-Tuning) ====================")
print(f"  Accuracy  : {baseline_results['eval_accuracy']:.4f}")
print(f"  F1        : {baseline_results['eval_f1']:.4f}")
print(f"  Precision : {baseline_results['eval_precision']:.4f}")
print(f"  Recall    : {baseline_results['eval_recall']:.4f}")
print(f"  Loss      : {baseline_results['eval_loss']:.4f}")

# ALSO SAVE FULL CLASSIFICATION REPORT ------------------------------------------------------------------
predictions = trainer.predict(test_ds)
preds = predictions.predictions.argmax(axis=-1)
labels = predictions.label_ids

print("\n==================== FULL CLASSIFICATION REPORT ====================")
print(classification_report(labels, preds, target_names=["Class 0", "Class 1"]))

# SAVE BASELINE RESULTS TO FILE -------------------------------------------------------------------------
results_df = pd.DataFrame([{
    "model": "Bio_ClinicalBERT (baseline, no fine-tuning)",
    "accuracy": baseline_results["eval_accuracy"],
    "f1": baseline_results["eval_f1"],
    "precision": baseline_results["eval_precision"],
    "recall": baseline_results["eval_recall"],
    "loss": baseline_results["eval_loss"],
    "test_set_size": len(test_df),
}])

output_path = "/scratch/user/charu7465/baseline_results/baseline_metrics.csv"
results_df.to_csv(output_path, index=False)
print(f"\nBaseline metrics saved to: {output_path}")
print("\nNow run your fine-tuning script and compare the final test metrics against these numbers!")