#!/bin/bash
#SBATCH --job-name=icd_pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/user/kiana.shen22/icd_logs/%j.out
#SBATCH --error=/scratch/user/kiana.shen22/icd_logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kiana.shen22@tamu.edu

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1
# If Grace says the above module does not exist, use:
# module load GCCcore/12.3.0 Python/3.11.3

export PROJECT_DIR=/scratch/user/kiana.shen22/FinalProject_421
export PY_LIBS=/scratch/user/kiana.shen22/python_libs
export HF_HOME=/scratch/user/kiana.shen22/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PY_LIBS:$PROJECT_DIR:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1

mkdir -p /scratch/user/kiana.shen22/icd_logs "$HF_HOME" "$PROJECT_DIR/predictions" "$PROJECT_DIR/checkpoints"

echo "Job started: $(date)"
echo "Node: $(hostname)"
python -c "import sys, torch; print(sys.executable); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
nvidia-smi || true

cd "$PROJECT_DIR"
python pipeline.py \
  --base_dir "$PROJECT_DIR" \
  --train "$PROJECT_DIR/train_data-text_and_labels.csv" \
  --test_dir "$PROJECT_DIR/test_files" \
  --manual "$PROJECT_DIR/manual_labeled_751.csv" \
  --model_name_or_path "/scratch/user/kiana.shen22/Bio_ClinicalBERT" \
  --model_cache "$HF_HOME" \
  --checkpoint_dir "$PROJECT_DIR/checkpoints" \
  --output_dir "$PROJECT_DIR/predictions" \
  --no_external

echo "Job finished: $(date)"
