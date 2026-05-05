#!/bin/bash
#SBATCH --job-name=cf_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cf_train_%j.out
#SBATCH --error=logs/cf_train_%j.err

set -euo pipefail
mkdir -p logs artifacts predictions hf_cache
module purge
module load GCCcore/12.3.0 Python
source /scratch/user/siyabhatkal2010/icd_venv/bin/activate

export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$PWD/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Point at the local Bio_ClinicalBERT copy you uploaded.
# (Download from https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
#  on a machine with internet, then upload to bio_clinical_bert/.)
export MODEL_DIR=${MODEL_DIR:-$PWD/bio_clinical_bert}
export MODEL_NAME=${MODEL_NAME:-emilyalsentzer/Bio_ClinicalBERT}

# CV + final training hyperparameters
export MAX_LENGTH=${MAX_LENGTH:-128}
export BATCH_SIZE=${BATCH_SIZE:-16}
export FOLDS=${FOLDS:-5}
export CV_EPOCHS=${CV_EPOCHS:-8}
export FINAL_EPOCHS=${FINAL_EPOCHS:-30}
export LR_BERT=${LR_BERT:-2e-5}
export LR_HEAD=${LR_HEAD:-1e-3}
export DROPOUT=${DROPOUT:-0.3}
export SEEDS=${SEEDS:-42,43,44,45}
export TRUNC_SIDE=${TRUNC_SIDE:-right}
export THRESHOLD_DEFAULT=${THRESHOLD_DEFAULT:-0.50}

python -u train.py
