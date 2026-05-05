# cf — Bio_ClinicalBERT pipeline

This is the second pipeline, implementing the architecture described in the
team writeup:

- **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) — BERT base, fully fine-tuned end-to-end.
- **Architecture:** BERT base → Dropout(0.3) → Linear(768 → 2)
- **Differential learning rates:** BERT layers 2e-5, classifier head 1e-3
- **Linear warmup (10%) + AdamW**, gradient clip 1.0, max_length 128
- **Training data:** 20 gold examples + ~700 weakly labeled MIMIC-III sentences
  calibrated to 200 pos / 500 neg (~29% positive, matching the test prior)
- **Strategy:** stratified 5-fold CV across seeds 42–45, pick best mean F1,
  retrain on the full set for 30 epochs with the best seed
- **Post-processing:** rule-based 1→0 overrides for unambiguous normal-finding
  language (no acute X, within normal limits, normal LVEF, no evidence of …
  unless strong-pathology keyword present, etc.)

## Directory layout

```
cf/
├── generate_data.py       # MIMIC-III note → calibrated training set
├── train.py               # CV + multi-seed + final retrain
├── predict.py             # predict + post-processing overrides
├── run_generate_grace.sh  # sbatch wrapper for generate_data.py (CPU)
├── run_train_grace.sh     # sbatch wrapper for train.py (GPU, ~30 min)
├── run_predict_grace.sh   # sbatch wrapper for predict.py (GPU, ~2 min)
├── requirements.txt
├── data/
│   ├── NOTEEVENTS.csv                       # MIMIC-III notes (you provide)
│   ├── train_data-text_and_labels.csv       # 20 gold examples
│   ├── test01_text_only.csv                 # autograder test sets
│   ├── test02_text_only.csv
│   ├── test03_text_only.csv
│   └── combined/combined_train.csv          # written by generate_data.py
├── bio_clinical_bert/                       # local copy of the HF weights (you provide)
├── artifacts/
│   ├── bert_model/                          # encoder + tokenizer + head.pt (written by train.py)
│   ├── threshold.json                       # written by train.py
│   └── cv_summary.json
├── predictions/                             # written by predict.py
└── logs/
```

## One-time setup

### 1. Download Bio_ClinicalBERT locally and upload to Grace

Grace has no outbound internet, so the model has to be cached.

On a machine with internet (your laptop):

```python
from transformers import AutoTokenizer, AutoModel
m = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
t = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
m.save_pretrained("bio_clinical_bert")
t.save_pretrained("bio_clinical_bert")
```

Then `scp -r bio_clinical_bert/ siyabhatkal2010@grace.hprc.tamu.edu:/scratch/user/siyabhatkal2010/pipeline/cf/`.

### 2. Copy data into cf/data/

```bash
cp ../icd_codable_pipeline/data/train_data-text_and_labels.csv data/
cp ../icd_codable_pipeline/data/test*_text_only.csv data/
# NOTEEVENTS.csv: copy from wherever you have it on Grace
```

### 3. Run the pipeline

```bash
sbatch run_generate_grace.sh   # ~5-10 min
# wait for: data/combined/combined_train.csv

sbatch run_train_grace.sh      # ~30-45 min: 5 folds * 4 seeds * 8 epochs + final retrain 30 epochs
# wait for: artifacts/bert_model/ + artifacts/threshold.json

sbatch run_predict_grace.sh    # ~2 min
# wait for: predictions/test*-pred.csv
```

## Tuning knobs

All env vars override the defaults — set them before `sbatch`:

```bash
# Faster CV pass (cuts training time in half):
CV_EPOCHS=4 FINAL_EPOCHS=20 sbatch run_train_grace.sh

# Larger or smaller training set:
WANT_POS=300 WANT_NEG=750 sbatch run_generate_grace.sh

# Disable post-processing overrides (debugging):
APPLY_OVERRIDES=0 sbatch run_predict_grace.sh
```

## Expected behavior

- generate_data.py: prints "Saved 720 rows to data/combined/combined_train.csv"
- train.py: prints per-fold F1, then "BEST SEED: 44 mean CV F1=0.812", then
  "Final holdout best threshold=0.XXX F1=0.XX", then "Training complete."
- predict.py: prints per-file pos_rate around 0.28–0.32 for test01/test02 and
  ~0.50 for test03.
