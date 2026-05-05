#!/bin/bash
#SBATCH --job-name=icd_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/icd_train_%j.out
#SBATCH --error=logs/icd_train_%j.err

set -euo pipefail
mkdir -p logs artifacts predictions hf_cache
module purge
module load GCCcore/12.3.0 Python
source /scratch/user/siyabhatkal2010/icd_venv/bin/activate

# Force HF to use the local cache only — Grace has no internet.
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$PWD/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Use the locally saved ClinicalBERT copy that v1 train.py created.
# bert_model/ contains config.json + pytorch_model.bin + tokenizer files.
export MODEL_DIR=${MODEL_DIR:-$PWD/bert_model}

export MODEL_NAME=${MODEL_NAME:-medicalai/ClinicalBERT}
export MAX_LENGTH=${MAX_LENGTH:-256}
export BATCH_SIZE=${BATCH_SIZE:-16}
export EPOCHS=${EPOCHS:-3}
export LR=${LR:-2e-5}

python -u train.py
