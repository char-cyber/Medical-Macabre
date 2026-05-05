"""
trainer.py — trains an ensemble of:
  1. Fine-tuned ClinicalBERT (medicalai/ClinicalBERT)
  2. Fine-tuned DistilBERT
  3. Fine-tuned BERT-base-uncased
  4. TF-IDF + Logistic Regression (fast, strong baseline)

Each BERT model is trained with:
  • Weighted cross-entropy (class imbalance)
  • FP16, TF32, fused AdamW (Grace A100 speed)
  • Early stopping on validation F1
  • Gradient checkpointing

Final ensemble uses soft voting (average predicted probabilities).
"""

import os, json
import numpy as np
import pandas as pd
import torch
import re
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)

USE_FP16 = torch.cuda.is_available()
MAX_LENGTH       = 256
NUM_EPOCHS       = 3
BATCH_SIZE       = 8
EVAL_BATCH_SIZE  = 16
LEARNING_RATE    = 3e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
EARLY_STOP_PAT   = 3
DATALOADER_WORKERS = 2
RANDOM_STATE     = 42
GRADIENT_ACCUMULATION_STEPS = 2

def extract_clinical_features(texts: list[str]) -> np.ndarray:
    """Add hand-crafted clinical features"""
    features = []
    
    # Clinical patterns
    icd9_pattern = re.compile(r'\b\d{3}\.\d{1,2}\b')
    icd10_pattern = re.compile(r'[A-Z][0-9]{2}\.[0-9]{1,2}')
    
    for text in texts:
        f = []
        # Length features
        f.append(len(text.split()))
        f.append(len(text))
        
        # ICD code presence
        f.append(1 if icd9_pattern.search(text) else 0)
        f.append(1 if icd10_pattern.search(text) else 0)
        
        # Section header importance (HPI, Assessment are more valuable)
        section_boost = 0
        if any(header in text.lower() for header in ['hpi', 'assessment', 'impression']):
            section_boost = 1
        f.append(section_boost)
        
        # Numerical values (labs, vitals)
        f.append(len(re.findall(r'\b\d+(?:\.\d+)?\b', text)))
        
        features.append(f)
    
    return np.array(features)
# =============================================================================
# WEIGHTED LOSS TRAINER
# =============================================================================
class WeightedLossTrainer(Trainer): #lets make this focal loss
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


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="binary", zero_division=0),
    }


# ----------------------- THRESHOLDS --------------------------------
def find_best_threshold(probs, labels):  
    """
    Grid-search threshold using validation probabilities.
    Uses macro F1 so over-predicting either class is punished.
    """
    thresholds = np.linspace(0.30, 0.85, 30)

    best_t = 0.5
    best_f1 = -1.0

    labels = np.array(labels)

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"  Best threshold: {best_t:.3f} | Val Macro F1: {best_f1:.4f}")
    return best_t, best_f1

class QuietCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(
            f"    Epoch {state.epoch:.1f} | "
            f"Loss={metrics.get('eval_loss',0):.4f} | "
            f"F1={metrics.get('eval_f1',0):.4f} | "
            f"Acc={metrics.get('eval_accuracy',0):.4f}"
        )


# =============================================================================
# SINGLE BERT FINE-TUNER
# =============================================================================
def fine_tune_bert(
    model_name: str,
    train_texts, train_labels,
    val_texts,   val_labels,
    checkpoint_dir: Path,
    model_cache: Path,
    device,
) -> dict:
    """
    Fine-tune one BERT-family model.
    Returns dict with keys: 'model', 'tokenizer', 'val_f1'
    """
    safe_name = model_name.replace("/", "_")
    out_dir    = checkpoint_dir / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Fine-tuning: {model_name}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(model_cache),
        local_files_only=os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    )

    # ── Tokenize ─────────────────────────────────────────────────────────────
    def tokenize(texts, labels_arr):
        ds = Dataset.from_dict({"text": list(texts), "labels": list(labels_arr)})
        ds = ds.map(
            lambda b: tokenizer(
                b["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            batched=True,
            num_proc=1,
        )
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    train_ds = tokenize(train_texts, train_labels)
    val_ds   = tokenize(val_texts,   val_labels)

    # ── Class weights ────────────────────────────────────────────────────────
    cw = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=np.array(train_labels)
    )
    cw_tensor = torch.tensor(cw, dtype=torch.float)

    # ── Model ────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        cache_dir=str(model_cache),
        local_files_only=os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    )
    model.gradient_checkpointing_enable()

    # ── Training args ─────────────────────────────────────────────────────────
    # Compatible with both newer and older Transformers versions on Grace.
    import inspect
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    arg_dict = dict(
        output_dir=str(out_dir),
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=USE_FP16,
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=True,
        group_by_length=True,
        optim="adamw_torch",
        report_to="none",
        disable_tqdm=True,
        logging_dir=str(checkpoint_dir / "logs"),
    )
    if "eval_strategy" in ta_params:
        arg_dict["eval_strategy"] = "epoch"
    else:
        arg_dict["evaluation_strategy"] = "epoch"
    if "tf32" in ta_params:
        arg_dict["tf32"] = True
    args = TrainingArguments(**arg_dict)

    trainer_kwargs = dict(
        class_weights=cw_tensor,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            QuietCallback(),
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PAT),
        ],
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedLossTrainer(**trainer_kwargs)
    trainer.train()
    val_results = trainer.evaluate()
    val_f1 = val_results.get("eval_f1", 0)
    print(f"  → {model_name} val F1: {val_f1:.4f}")

    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"  Saved best model/tokenizer → {final_dir}")

    return {
        "model":     trainer.model,
        "tokenizer": tokenizer,
        "val_f1":    val_f1,
        "name":      model_name,
    }


# =============================================================================
# LOGISTIC REGRESSION COMPONENT
# =============================================================================
def train_logistic_regression(train_texts, train_labels, checkpoint_dir: Path):
    """TF-IDF + LogReg — fast and surprisingly strong on clinical text."""
    print("\n  Training TF-IDF + Logistic Regression...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=50_000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(train_texts, train_labels)

    # Save
    lr_path = checkpoint_dir / "logreg_pipeline.joblib"
    joblib.dump(pipe, lr_path)
    print(f"  LR pipeline saved → {lr_path}")
    return pipe


# =============================================================================
# ENSEMBLE WRAPPER
# =============================================================================
class Ensemble:
    """
    Wraps a list of fine-tuned BERT models + a LR pipeline.
    predict_proba(texts) → np.ndarray shape (N, 2)
    predict(texts)       → np.ndarray shape (N,)  binary 0/1
    """
    def __init__(self, bert_models: list, lr_pipeline, device, bert_weight=0.75):
        """
        bert_weight : fraction of final vote from BERT ensemble (rest from LR)
        """
        self.bert_models  = bert_models   # list of dicts {model, tokenizer, name}
        self.lr_pipeline  = lr_pipeline
        self.device       = device
        self.bert_weight  = bert_weight
        self.lr_weight    = 1.0 - bert_weight

    def _bert_proba(self, texts: list[str]) -> np.ndarray:
        """Average softmax probabilities across all BERT models."""
        all_probs = []
        for m in self.bert_models:
            model     = m["model"].to(self.device).eval()
            tokenizer = m["tokenizer"]
            probs     = []

            # Batch inference
            batch_size = 64
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    logits = model(**enc).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

            all_probs.append(np.vstack(probs))

        return np.mean(all_probs, axis=0)   # shape (N, 2)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        bert_p = self._bert_proba(texts)                            # (N, 2)
        lr_p   = self.lr_pipeline.predict_proba(texts)
        clinical_features = extract_clinical_features(texts)
        # Simple rule-based boost for high-confidence clinical features
        feature_boost = np.clip(clinical_features[:, 2] * 0.2, 0, 0.3) 
        # Weighted combination
        combined = (0.6 * bert_p + 0.3 * lr_p)
        
        # Apply feature boost to class 1 probability
        combined[:, 1] += feature_boost
        combined = combined / combined.sum(axis=1, keepdims=True)
        
        return combined            # (N, 2)
        #return self.bert_weight * bert_p + self.lr_weight * lr_p    # (N, 2)

    def predict(self, texts: list[str], threshold: float = 0.35) -> np.ndarray:
        return (self.predict_proba(texts)[:, 1] > threshold).astype(int)

    # ADDED
    def predict_proba_class1(self, texts: list[str]) -> np.ndarray:
        """
        Returns P(class=1) for each sentence
        """
        probs = self.predict_proba(texts)  # shape (N, 2)
        return probs[:, 1]

# =============================================================================
# MAIN ENTRY
# =============================================================================
def train_ensemble(
    train_df: pd.DataFrame,
    model_names: list[str],
    device,
    checkpoint_dir: Path,
    model_cache: Path,
) -> Ensemble:
    """
    Train all components and return an Ensemble object.

    Uses `label_binary` column (0/1) for training — negated class (-1) maps to 0.
    """
    texts  = train_df["text"].tolist()
    labels = train_df["label_binary"].tolist()

    # ── Train / val split ────────────────────────────────────────────────────
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    print(f"  Train: {len(train_texts)}  Val: {len(val_texts)}")

    # ── Fine-tune each BERT ───────────────────────────────────────────────────
    bert_models = []
    for mname in model_names:
        result = fine_tune_bert(
            mname,
            train_texts, train_labels,
            val_texts,   val_labels,
            checkpoint_dir,
            model_cache,
            device,
        )
        bert_models.append(result)

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_pipe = train_logistic_regression(train_texts, train_labels, checkpoint_dir)

    # ── Evaluate ensemble on val set ─────────────────────────────────────────
    print("\n  Evaluating ensemble on validation set...")
    ensemble = Ensemble(bert_models, lr_pipe, device)

    val_probs = ensemble.predict_proba_class1(val_texts)
    best_threshold, best_macro_f1 = find_best_threshold(val_probs, val_labels)
    ensemble.threshold = best_threshold

    val_preds = (val_probs >= ensemble.threshold).astype(int)

    val_f1_binary = f1_score(val_labels, val_preds, average="binary", zero_division=0)
    val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    val_acc = accuracy_score(val_labels, val_preds)

    print(f"\n  Ensemble Val Binary F1 : {val_f1_binary:.4f}")
    print(f"  Ensemble Val Macro F1  : {val_f1_macro:.4f}")
    print(f"  Ensemble Val Acc       : {val_acc:.4f}")
    print(f"  Saved threshold        : {ensemble.threshold:.3f}")

    print("\n" + classification_report(
        val_labels,
        val_preds,
        target_names=["Not useful (0)", "Useful (1)"],
        zero_division=0,
    ))

    return ensemble
