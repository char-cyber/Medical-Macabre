#!/bin/bash
#SBATCH --job-name=icd_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/icd_train_%j.out
#SBATCH --error=logs/icd_train_%j.err

set -euo pipefail
mkdir -p logs artifacts predictions hf_cache
module purge
module load GCCcore/12.3.0 Python
source /scratch/user/$USER/csce421_icd_venv/bin/activate
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$PWD/hf_cache
export TOKENIZERS_PARALLELISM=false
python -u train.py \
  --train_csv data/train_data-text_and_labels.csv \
  --manual_csv data/manual_751.csv \
  --weak_notes_glob "data/mimic_notes/*.csv" \
  --weak_max 3000 \
  --models medicalai/ClinicalBERT distilbert-base-uncased bert-base-uncased \
  --cache_dir hf_cache \
  --out artifacts/icd_ensemble.joblib \
  --batch_size 16
