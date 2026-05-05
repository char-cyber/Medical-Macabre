#!/bin/bash
#SBATCH --job-name=icd_predict
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@tamu.edu

cd /scratch/user/charu7465

MODEL_DIR="/scratch/user/charu7465/baseline_results/clinicalbert"
PREDICTIONS="/scratch/user/charu7465/predictions"
export PYTHONPATH="/scratch/user/charu7465/pypackages:$PYTHONPATH"
export TRANSFORMERS_CACHE="/scratch/user/charu7465/hf_cache"

mkdir -p "$PREDICTIONS"

echo "=== Running predictions ==="
python predict.py \
    --model_dir  "$MODEL_DIR" \
    --test_files test01_text_only.csv \
                 test02_text_only.csv \
                 test03_text_only.csv \
    --output_dir "$PREDICTIONS" \
    --max_len    128 \
    --batch_size 32 \
    --threshold  0.3

echo ""
echo "=== Output files ==="
ls -lh "$PREDICTIONS"