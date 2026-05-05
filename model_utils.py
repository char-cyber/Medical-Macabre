import os
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

MODEL_NAMES = ["medicalai/ClinicalBERT", "distilbert-base-uncased", "bert-base-uncased"]


def label_to_binary(x):
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"1", "yes", "y", "true", "useful", "codable", "icd", "positive"}:
            return 1
        if v in {"0", "no", "n", "false", "not useful", "not_useful", "negative", "-1"}:
            return 0
    try:
        return 1 if int(float(x)) == 1 else 0
    except Exception:
        return 0


class BertEmbedder:
    def __init__(self, model_name: str, cache_dir: str = "./hf_cache", max_length: int = 128):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 16):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            mask = toks["attention_mask"].unsqueeze(-1)
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            vecs = (summed / counts).detach().cpu().numpy()
            all_vecs.append(vecs)
        return np.vstack(all_vecs)


def train_lr_on_embeddings(X, y):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)),
    ])
    clf.fit(X, y)
    return clf


def save_ensemble(path, ensemble):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(ensemble, path)


def load_ensemble(path):
    return joblib.load(path)
