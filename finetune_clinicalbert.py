import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
import json

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
# CONFIG
# -----------------------------------------------------------------------------------------
FILE_NAME           = "data_test.csv"
MODEL_NAME          = "/scratch/user/charu7465/Bio_ClinicalBERT_new"
BASE_OUTPUT_DIR     = "/scratch/user/charu7465/clinicalbert_finetuned"
LOG_DIR             = "/scratch/user/charu7465/logs"
RESULTS_PATH        = "/scratch/user/charu7465/clinicalbert_finetuned/all_results.json"

MAX_LENGTH          = 128
N_FOLDS             = 5
RANDOM_STATE        = 42
EARLY_STOP_PATIENCE = 3
NUM_EPOCHS          = 15
WEIGHT_DECAY        = 0.01

LEARNING_RATES      = [2e-5, 3e-5]
BATCH_SIZES         = [8, 16]

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------------------
# STEP 1 — LOAD, CLEAN, AND PREPROCESS ONCE
# Everything downstream uses indices into this single processed dataset
# -----------------------------------------------------------------------------------------
df = pd.read_csv(FILE_NAME)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df = df.dropna(subset=["text", "label"])
df["text"]  = df["text"].astype(str)
df["label"] = df["label"].astype(int)

print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Class imbalance ratio: {df['label'].value_counts()[0] / df['label'].value_counts()[1]:.2f}:1\n")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# -----------------------------------------------------------------------------------------
# STEP 2 — TOKENIZE THE ENTIRE DATASET ONCE
# Subsets are created by slicing indices — no re-tokenization anywhere
# -----------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

print("Tokenizing full dataset once...")
full_ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
full_ds = full_ds.map(
    lambda batch: tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    ),
    batched=True
)
full_ds = full_ds.rename_column("label", "labels")
full_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
print(f"Tokenization complete. Dataset size: {len(full_ds)}\n")

def get_split(indices):
    """Return a torch-ready dataset from a list of integer indices."""
    return full_ds.select(indices)

# -----------------------------------------------------------------------------------------
# STEP 3 — CLASS WEIGHTS (computed once from full label array)
# -----------------------------------------------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=df["label"].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"Class weights:")
print(f"  Class 0: {class_weights[0]:.4f}")
print(f"  Class 1: {class_weights[1]:.4f}  <-- higher = model penalized more for missing Class 1\n")

# -----------------------------------------------------------------------------------------
# STEP 4 — DEFINE SPLITS ONCE
# HP search and final test use the same held-out test set
# -----------------------------------------------------------------------------------------
all_idx   = np.arange(len(df))
train_val_idx, test_idx = train_test_split(
    all_idx,
    test_size=0.10,              # 10% held out as final test set
    stratify=df["label"].values,
    random_state=RANDOM_STATE
)
hp_train_idx, hp_val_idx = train_test_split(
    train_val_idx,
    test_size=0.15,              # ~15% of train_val used for HP search validation
    stratify=df["label"].values[train_val_idx],
    random_state=RANDOM_STATE
)

print(f"Split sizes:")
print(f"  HP train : {len(hp_train_idx)}")
print(f"  HP val   : {len(hp_val_idx)}")
print(f"  Test     : {len(test_idx)} (held out — never seen during training or HP search)\n")

hp_train_ds = get_split(hp_train_idx)
hp_val_ds   = get_split(hp_val_idx)
test_ds     = get_split(test_idx)

# -----------------------------------------------------------------------------------------
# WEIGHTED LOSS TRAINER — defined once, reused across all runs
# -----------------------------------------------------------------------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )(logits, labels)
        return (loss, outputs) if return_outputs else loss

# -----------------------------------------------------------------------------------------
# METRICS — defined once
# -----------------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    p1, r1, f1c, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[1]
    )
    return {
        "accuracy":         acc,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "class1_precision": float(p1[0]),
        "class1_recall":    float(r1[0]),   # key metric — was 0.22 at baseline
        "class1_f1":        float(f1c[0]),
    }

class PrettyMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(
            f"  Epoch {state.epoch:.1f} | "
            f"Loss: {metrics.get('eval_loss', 0):.4f} | "
            f"F1: {metrics.get('eval_f1', 0):.4f} | "
            f"Acc: {metrics.get('eval_accuracy', 0):.4f} | "
            f"C1 Recall: {metrics.get('eval_class1_recall', 0):.4f}"
        )

# -----------------------------------------------------------------------------------------
# TRAINING FUNCTION
# Model is the only thing re-initialized per run (fresh weights required each time)
# Everything else (tokenizer, dataset, weights, metrics) is reused from outer scope
# -----------------------------------------------------------------------------------------
def run_training(train_ds, val_ds, lr, batch_size, output_dir):
    # Fresh model each run — this is intentional and necessary
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        local_files_only=True
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,
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
    results = trainer.evaluate()
    return trainer, results

# -----------------------------------------------------------------------------------------
# STEP 5 — HYPERPARAMETER SEARCH
# Runs on the fixed HP train/val split — not the test set
# -----------------------------------------------------------------------------------------
print("=" * 60)
print("STEP 5: HYPERPARAMETER SEARCH")
print(f"Grid: lr={LEARNING_RATES}, batch_size={BATCH_SIZES}")
print("=" * 60)

best_f1    = -1
best_lr    = None
best_batch = None
hp_results = []

for lr in LEARNING_RATES:
    for bs in BATCH_SIZES:
        print(f"\n--- lr={lr}, batch_size={bs} ---")
        out_dir = f"{BASE_OUTPUT_DIR}/hp_search/lr{lr}_bs{bs}"
        os.makedirs(out_dir, exist_ok=True)

        _, results = run_training(hp_train_ds, hp_val_ds, lr, bs, out_dir)

        f1  = results.get("eval_f1", 0)
        acc = results.get("eval_accuracy", 0)
        c1r = results.get("eval_class1_recall", 0)
        print(f"  --> F1: {f1:.4f} | Acc: {acc:.4f} | C1 Recall: {c1r:.4f}")

        hp_results.append({"lr": lr, "batch_size": bs, "val_f1": f1,
                            "val_accuracy": acc, "class1_recall": c1r})

        if f1 > best_f1:
            best_f1, best_lr, best_batch = f1, lr, bs

print(f"\nBEST: lr={best_lr}, batch_size={best_batch}, F1={best_f1:.4f}\n")

# -----------------------------------------------------------------------------------------
# STEP 6 — K-FOLD CROSS VALIDATION on train_val pool (test set excluded)
# Indices sliced from the already-tokenized full_ds — no re-tokenization
# -----------------------------------------------------------------------------------------
print("=" * 60)
print(f"STEP 6: {N_FOLDS}-FOLD CROSS VALIDATION")
print(f"Using best: lr={best_lr}, batch_size={best_batch}")
print("=" * 60)

skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_scores = []

kfold_labels = df["label"].values[train_val_idx]

for fold, (rel_train, rel_val) in enumerate(skf.split(train_val_idx, kfold_labels), 1):
    print(f"\n--- Fold {fold}/{N_FOLDS} ---")

    # Map relative indices back to absolute dataset indices
    abs_train = train_val_idx[rel_train]
    abs_val   = train_val_idx[rel_val]

    fold_train_ds = get_split(abs_train)
    fold_val_ds   = get_split(abs_val)

    out_dir = f"{BASE_OUTPUT_DIR}/kfold/fold_{fold}"
    os.makedirs(out_dir, exist_ok=True)

    _, results = run_training(fold_train_ds, fold_val_ds, best_lr, best_batch, out_dir)

    fold_scores.append({
        "fold":          fold,
        "f1":            results.get("eval_f1", 0),
        "accuracy":      results.get("eval_accuracy", 0),
        "precision":     results.get("eval_precision", 0),
        "recall":        results.get("eval_recall", 0),
        "class1_recall": results.get("eval_class1_recall", 0),
        "class1_f1":     results.get("eval_class1_f1", 0),
    })
    print(f"  Fold {fold} → F1: {fold_scores[-1]['f1']:.4f} | C1 Recall: {fold_scores[-1]['class1_recall']:.4f}")

f1_scores  = [s["f1"] for s in fold_scores]
acc_scores = [s["accuracy"] for s in fold_scores]
c1r_scores = [s["class1_recall"] for s in fold_scores]

print(f"\n{'=' * 60}")
print(f"K-FOLD SUMMARY ({N_FOLDS} folds)")
print(f"  Mean F1       : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}  (baseline: 0.2353)")
print(f"  Mean Accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}  (baseline: 0.4583)")
print(f"  Mean C1 Recall: {np.mean(c1r_scores):.4f} ± {np.std(c1r_scores):.4f}  (baseline: 0.2222)")
print(f"{'=' * 60}\n")

# -----------------------------------------------------------------------------------------
# STEP 7 — FINAL MODEL: train on all train_val data, evaluate on held-out test set
# -----------------------------------------------------------------------------------------
print("=" * 60)
print("STEP 7: FINAL MODEL TRAINING & TEST EVALUATION")
print("=" * 60)

final_train_ds = get_split(train_val_idx)   # all non-test data
# test_ds already defined above from test_idx

final_out_dir = f"{BASE_OUTPUT_DIR}/final_model"
os.makedirs(final_out_dir, exist_ok=True)

final_trainer, _ = run_training(
    final_train_ds, test_ds, best_lr, best_batch, final_out_dir
)

test_results = final_trainer.evaluate(test_ds)

print(f"\n{'=' * 60}")
print(f"FINAL TEST RESULTS vs BASELINE")
print(f"{'Metric':<22} {'Baseline':>10} {'Fine-Tuned':>12}")
print(f"{'-' * 46}")
print(f"{'Accuracy':<22} {'0.4583':>10} {test_results['eval_accuracy']:>12.4f}")
print(f"{'F1':<22} {'0.2353':>10} {test_results['eval_f1']:>12.4f}")
print(f"{'Precision':<22} {'0.2500':>10} {test_results['eval_precision']:>12.4f}")
print(f"{'Recall (Class 1)':<22} {'0.2222':>10} {test_results['eval_recall']:>12.4f}")
print(f"{'=' * 60}")

predictions = final_trainer.predict(test_ds)
preds  = predictions.predictions.argmax(axis=-1)
labels = predictions.label_ids
print("\nFULL CLASSIFICATION REPORT:")
print(classification_report(labels, preds, target_names=["Class 0 (Negative)", "Class 1 (Positive)"]))

# -----------------------------------------------------------------------------------------
# SAVE ALL RESULTS
# -----------------------------------------------------------------------------------------
all_results = {
    "baseline_results": {
        "accuracy": 0.4583, "f1": 0.2353,
        "precision": 0.25,  "recall": 0.2222
    },
    "best_hyperparameters": {
        "learning_rate": best_lr,
        "batch_size":    best_batch,
        "best_val_f1":   best_f1
    },
    "hyperparameter_search": hp_results,
    "kfold_results":   fold_scores,
    "kfold_summary": {
        "mean_f1":            float(np.mean(f1_scores)),
        "std_f1":             float(np.std(f1_scores)),
        "mean_accuracy":      float(np.mean(acc_scores)),
        "std_accuracy":       float(np.std(acc_scores)),
        "mean_class1_recall": float(np.mean(c1r_scores)),
        "std_class1_recall":  float(np.std(c1r_scores)),
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
save_path = f"{BASE_OUTPUT_DIR}/final_model/saved_model"
os.makedirs(save_path, exist_ok=True)
final_trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Final model saved to: {save_path}")
print("\nDone!")