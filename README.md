# ICD codable sentence classifier

This project trains a GPU based ensemble for predicting whether a clinical sentence contains ICD codable information.

## Main idea

1. Clean and split notes into sections with regex.
2. Split each section into sentences with SciSpacy, falling back to a spaCy sentencizer.
3. Convert labels to binary. A manual label of -1 is treated as a strong negative.
4. Encode sentences with ClinicalBERT, DistilBERT, and BERT.
5. Train one logistic regression classifier per embedding model.
6. Average model probabilities and tune the decision threshold on validation F1.
7. Predict every `test*_text_only.csv` file and write one `test##-pred.csv` file.

## Expected folders

```text
data/
  train_data-text_and_labels.csv
  manual_751.csv
  test1_text_only.csv
  test2_text_only.csv
  mimic_notes/
    NOTEEVENTS_sample.csv
```

Manual labels should have a sentence or text column and a label column. Labels can be 1, 0, or -1.

## Run

```bash
sbatch run_train_grace.sh
sbatch run_predict_grace.sh
```

Outputs are written to `predictions/`.
