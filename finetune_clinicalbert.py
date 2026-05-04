import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
import json
#from splitting_bert import *

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Starting enhanced fine-tuning...\n")

# -----------------------------------------------------------------------------------------
# CONFIG — edit these paths/settings as needed
# -----------------------------------------------------------------------------------------
FILE_NAME        = "train_labeled_clean.csv"
MODEL_NAME       = "/scratch/user/charu7465/Bio_ClinicalBERT_new"
BASE_OUTPUT_DIR  = "/scratch/user/charu7465/clinicalbert_finetuned"
LOG_DIR          = "/scratch/user/charu7465/logs"
RESULTS_PATH     = "/scratch/user/charu7465/clinicalbert_finetuned/all_results.json"

MAX_LENGTH       = 128
N_FOLDS          = 2       # for k-fold cross validation, maybe change to 5
RANDOM_STATE     = 42

# Hyperparameter grid — all combinations will be tried on fold 1 to find the best,
# then the best combo is used for the full k-fold run
LEARNING_RATES   = [2e-5] #, 3e-5, 5e-5
BATCH_SIZES      = [16, 32]
NUM_EPOCHS       = 10      # max epochs (early stopping may stop sooner)
WEIGHT_DECAY     = 0.01
EARLY_STOP_PATIENCE = 3   # stop if no F1 improvement for this many epochs

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------------------
# LOAD & CLEAN DATA
# -----------------------------------------------------------------------------------------
df = pd.read_csv(FILE_NAME)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df = df.dropna(subset=["text", "label"])
df["text"]  = df["text"].astype(str)
df["label"] = df["label"].astype(int)

print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

df["text"] = df["text"]

# -----------------------------------------------------------------------------------------
# CLASS WEIGHTS (for imbalanced data)
# -----------------------------------------------------------------------------------------
classes      = np.array([0, 1])
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["label"].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"Class weights: Class 0 = {class_weights[0]:.4f}, Class 1 = {class_weights[1]:.4f}\n")

# -----------------------------------------------------------------------------------------
# TOKENIZER
# -----------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

def make_dataset(dataframe):
    ds = Dataset.from_pandas(dataframe[["text", "label"]].reset_index(drop=True))
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

# -----------------------------------------------------------------------------------------
# WEIGHTED LOSS TRAINER (handles class imbalance)
# -----------------------------------------------------------------------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device if torch.cuda.is_available() else "cpu")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        # move weights to same device as logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# -----------------------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -----------------------------------------------------------------------------------------
# PRETTY LOGGING CALLBACK
# -----------------------------------------------------------------------------------------
class PrettyMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(
            f"  Epoch {state.epoch:.1f} | "
            f"Loss: {metrics.get('eval_loss', 0):.4f} | "
            f"Acc: {metrics.get('eval_accuracy', 0):.4f} | "
            f"F1: {metrics.get('eval_f1', 0):.4f} | "
            f"Prec: {metrics.get('eval_precision', 0):.4f} | "
            f"Rec: {metrics.get('eval_recall', 0):.4f}"
        )

# -----------------------------------------------------------------------------------------
# TRAINING FUNCTION
# -----------------------------------------------------------------------------------------
def run_training(train_ds, val_ds, lr, batch_size, output_dir, fold_label=""):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        local_files_only=True
    )

    args = TrainingArguments(
        output_dir=output_dir,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        logging_dir=LOG_DIR,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            PrettyMetricsCallback(),
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)
        ],
    )

    trainer.train()
    val_results = trainer.evaluate()
    return trainer, val_results

# -----------------------------------------------------------------------------------------
# STEP 1 — HYPERPARAMETER SEARCH (on a single 80/20 split to keep it fast)
# -----------------------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: HYPERPARAMETER SEARCH")
print("=" * 60)

train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df["label"], random_state=RANDOM_STATE
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE
)

hp_train_ds = make_dataset(train_df)
hp_val_ds   = make_dataset(val_df)

best_f1     = -1
best_lr     = None
best_batch  = None
hp_results  = []

for lr in LEARNING_RATES:
    for bs in BATCH_SIZES:
        combo_label = f"lr{lr}_bs{bs}"
        print(f"\n--- Trying: lr={lr}, batch_size={bs} ---")
        out_dir = f"{BASE_OUTPUT_DIR}/hp_search/{combo_label}"
        os.makedirs(out_dir, exist_ok=True)

        _, results = run_training(hp_train_ds, hp_val_ds, lr, bs, out_dir)

        f1  = results.get("eval_f1", 0)
        acc = results.get("eval_accuracy", 0)
        print(f"  --> Val F1: {f1:.4f} | Val Acc: {acc:.4f}")

        hp_results.append({"lr": lr, "batch_size": bs, "val_f1": f1, "val_accuracy": acc})

        if f1 > best_f1:
            best_f1    = f1
            best_lr    = lr
            best_batch = bs

print(f"\n{'=' * 60}")
print(f"BEST HYPERPARAMETERS: lr={best_lr}, batch_size={best_batch}, F1={best_f1:.4f}")
print(f"{'=' * 60}\n")

# -----------------------------------------------------------------------------------------
# STEP 2 — K-FOLD CROSS VALIDATION with best hyperparameters
# -----------------------------------------------------------------------------------------
print("=" * 60)
print(f"STEP 2: {N_FOLDS}-FOLD CROSS VALIDATION")
print(f"Using: lr={best_lr}, batch_size={best_batch}")
print("=" * 60)

skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_scores = []

X = df["text"].values
y = df["label"].values

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n--- Fold {fold}/{N_FOLDS} ---")

    fold_train_df = df.iloc[train_idx]
    fold_val_df   = df.iloc[val_idx]

    fold_train_ds = make_dataset(fold_train_df)
    fold_val_ds   = make_dataset(fold_val_df)

    out_dir = f"{BASE_OUTPUT_DIR}/kfold/fold_{fold}"
    os.makedirs(out_dir, exist_ok=True)

    _, results = run_training(
        fold_train_ds, fold_val_ds,
        best_lr, best_batch,
        out_dir,
        fold_label=f"Fold {fold}"
    )

    fold_f1  = results.get("eval_f1", 0)
    fold_acc = results.get("eval_accuracy", 0)
    fold_scores.append({
        "fold": fold,
        "f1": fold_f1,
        "accuracy": fold_acc,
        "precision": results.get("eval_precision", 0),
        "recall": results.get("eval_recall", 0),
    })
    print(f"  Fold {fold} Results → F1: {fold_f1:.4f} | Acc: {fold_acc:.4f}")

# K-fold summary
f1_scores  = [s["f1"] for s in fold_scores]
acc_scores = [s["accuracy"] for s in fold_scores]

print(f"\n{'=' * 60}")
print(f"K-FOLD SUMMARY ({N_FOLDS} folds)")
print(f"  Mean F1       : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"  Mean Accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"{'=' * 60}\n")

# -----------------------------------------------------------------------------------------
# STEP 3 — FINAL MODEL: train on full train set, evaluate on held-out test set
# -----------------------------------------------------------------------------------------
print("=" * 60)
print("STEP 3: FINAL MODEL TRAINING & TEST EVALUATION")
print("=" * 60)

# Use the same test split from the HP search step as truly held-out data
final_train_df = df[~df.index.isin(test_df.index)]  # everything except test
final_train_ds = make_dataset(final_train_df)
final_test_ds  = make_dataset(test_df)

final_out_dir = f"{BASE_OUTPUT_DIR}/final_model"
os.makedirs(final_out_dir, exist_ok=True)

final_trainer, _ = run_training(
    final_train_ds, final_test_ds,
    best_lr, best_batch,
    final_out_dir
)

print("\nRunning final evaluation on held-out test set...")
test_results = final_trainer.evaluate(final_test_ds)

print(f"\n{'=' * 60}")
print("FINAL TEST RESULTS")
print(f"  Accuracy  : {test_results['eval_accuracy']:.4f}")
print(f"  F1        : {test_results['eval_f1']:.4f}")
print(f"  Precision : {test_results['eval_precision']:.4f}")
print(f"  Recall    : {test_results['eval_recall']:.4f}")
print(f"  Loss      : {test_results['eval_loss']:.4f}")
print(f"{'=' * 60}")

# Full classification report
predictions = final_trainer.predict(final_test_ds)
preds  = predictions.predictions.argmax(axis=-1)
labels = predictions.label_ids
print("\nFULL CLASSIFICATION REPORT:")
print(classification_report(labels, preds, target_names=["Class 0 (Negative)", "Class 1 (Positive)"]))

# -----------------------------------------------------------------------------------------
# SAVE ALL RESULTS TO JSON
# -----------------------------------------------------------------------------------------
all_results = {
    "best_hyperparameters": {
        "learning_rate": best_lr,
        "batch_size": best_batch,
        "best_val_f1": best_f1,
    },
    "hyperparameter_search": hp_results,
    "kfold_results": fold_scores,
    "kfold_summary": {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "mean_accuracy": float(np.mean(acc_scores)),
        "std_accuracy": float(np.std(acc_scores)),
    },
    "final_test_results": {
        "accuracy":  test_results["eval_accuracy"],
        "f1":        test_results["eval_f1"],
        "precision": test_results["eval_precision"],
        "recall":    test_results["eval_recall"],
        "loss":      test_results["eval_loss"],
    }
}

with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to: {RESULTS_PATH}")

# -----------------------------------------------------------------------------------------
# SAVE FINAL MODEL
# -----------------------------------------------------------------------------------------
final_trainer.save_model(f"{BASE_OUTPUT_DIR}/final_model/saved_model")
tokenizer.save_pretrained(f"{BASE_OUTPUT_DIR}/final_model/saved_model")
print(f"Final model saved to: {BASE_OUTPUT_DIR}/final_model/saved_model")
print("\nDone!")
