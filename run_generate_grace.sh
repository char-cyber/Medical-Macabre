#!/bin/bash
#SBATCH --job-name=cf_gen
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cf_gen_%j.out
#SBATCH --error=logs/cf_gen_%j.err

set -euo pipefail
mkdir -p logs data/combined
module purge
module load GCCcore/12.3.0 Python
source /scratch/user/siyabhatkal2010/icd_venv/bin/activate

export NOTES_CSV=${NOTES_CSV:-data/NOTEEVENTS.csv}
export N_NOTES=${N_NOTES:-8000}
export WANT_POS=${WANT_POS:-200}
export WANT_NEG=${WANT_NEG:-500}
export GOLD_CSV=${GOLD_CSV:-data/train_data-text_and_labels.csv}
export OUT_DIR=${OUT_DIR:-data/combined}
export SEED=${SEED:-42}

python -u generate_data.py
