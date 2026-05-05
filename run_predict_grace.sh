#!/bin/bash
#SBATCH --job-name=icd_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/icd_predict_%j.out
#SBATCH --error=logs/icd_predict_%j.err

set -euo pipefail
mkdir -p logs predictions hf_cache
module purge
module load GCCcore/12.3.0 Python
source /scratch/user/$USER/csce421_icd_venv/bin/activate
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$PWD/hf_cache
export TOKENIZERS_PARALLELISM=false
python -u predict.py \
  --model artifacts/icd_ensemble.joblib \
  --test_glob "data/test*_text_only.csv" \
  --out_dir predictions \
  --cache_dir hf_cache \
  --batch_size 16
