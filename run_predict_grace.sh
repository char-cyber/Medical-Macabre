#!/bin/bash
#SBATCH --job-name=icd_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/icd_predict_%j.out
#SBATCH --error=logs/icd_predict_%j.err

set -euo pipefail
mkdir -p logs predictions hf_cache
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

python -u predict.py
