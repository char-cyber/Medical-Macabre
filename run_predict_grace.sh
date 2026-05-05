#!/bin/bash
#SBATCH --job-name=cf_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=logs/cf_predict_%j.out
#SBATCH --error=logs/cf_predict_%j.err

set -euo pipefail
mkdir -p logs predictions
module purge
module load GCCcore/12.3.0 Python
source /scratch/user/siyabhatkal2010/icd_venv/bin/activate

export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$PWD/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export MODEL_DIR=${MODEL_DIR:-$PWD/artifacts/bert_model}
export THRESHOLD_PATH=${THRESHOLD_PATH:-$PWD/artifacts/threshold.json}
export TEST_GLOB=${TEST_GLOB:-data/test*_text_only.csv}
export OUT_DIR=${OUT_DIR:-predictions}
export BATCH_SIZE=${BATCH_SIZE:-32}
export APPLY_OVERRIDES=${APPLY_OVERRIDES:-1}

python -u predict.py
