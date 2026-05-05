#!/bin/bash
#SBATCH --job-name=icd_predict_only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/kiana.shen22/icd_logs/predict_%j.out
#SBATCH --error=/scratch/user/kiana.shen22/icd_logs/predict_%j.err
#SBATCH --exclude=g006

echo "Job started: $(date)"
echo "Node: $(hostname)"

# Load Grace GPU PyTorch stack
module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1

# Project paths
export PROJECT_DIR=/scratch/user/kiana.shen22/icd_project/grace-claude-pipeline
export PY_LIBS=/scratch/user/kiana.shen22/python_libs
export HF_HOME=/scratch/user/kiana.shen22/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Use your scratch libraries and project files
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PY_LIBS:$PROJECT_DIR:$PYTHONPATH

# Offline Hugging Face mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p /scratch/user/kiana.shen22/icd_logs

cd $PROJECT_DIR

echo "Python:"
which python

echo "Torch/CUDA check:"
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda devices', torch.cuda.device_count())"

nvidia-smi

echo "Running predict_only.py..."
python predict_only.py

echo "Job finished: $(date)"
