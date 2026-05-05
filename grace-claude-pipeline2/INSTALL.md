# Grace-ready setup for Kiana's ICD / ClinicalBERT pipeline

These files are configured for:

```bash
/scratch/user/kiana.shen22/FinalProject_421
/scratch/user/kiana.shen22/python_libs
/scratch/user/kiana.shen22/hf_cache
/scratch/user/kiana.shen22/Bio_ClinicalBERT
```

## 1. Copy the files to Grace

From your local folder that contains this edited code:

```bash
scp *.py run_pipeline.slurm kiana.shen22@grace.hprc.tamu.edu:/home/kiana.shen22/FinalProject_421/
```

Then on Grace:

```bash
mkdir -p /scratch/user/kiana.shen22/FinalProject_421
cp /home/kiana.shen22/FinalProject_421/*.py /scratch/user/kiana.shen22/FinalProject_421/
cp /home/kiana.shen22/FinalProject_421/run_pipeline.slurm /scratch/user/kiana.shen22/FinalProject_421/
```

## 2. Put your data in the expected places

```bash
cd /scratch/user/kiana.shen22/FinalProject_421
mkdir -p test_files predictions checkpoints

# Needed:
# train_data-text_and_labels.csv
# test_files/test01_text_only.csv, test02_text_only.csv, ...

# Optional:
# manual_labeled_751.csv
```

If your test files are currently in the main project folder:

```bash
mv test*_text_only.csv test_files/
```

## 3. Load the Grace Python/PyTorch stack

Use the same style you have been using on Grace:

```bash
module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1
```

If that exact module is unavailable, try:

```bash
module purge
module load GCCcore/12.3.0 Python/3.11.3
```

## 4. Use your scratch Python libraries

```bash
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
```

## 5. Check that your ClinicalBERT folder exists

```bash
ls /scratch/user/kiana.shen22/Bio_ClinicalBERT
```

If your already-trained classifier is somewhere else, use that path instead of `/scratch/user/kiana.shen22/Bio_ClinicalBERT` in `run_pipeline.slurm` and the `--model_name_or_path` command.

For example:

```bash
--model_name_or_path /scratch/user/kiana.shen22/clinicalbert_icd_classifier
```

## 6. Submit the GPU job

```bash
cd /scratch/user/kiana.shen22/FinalProject_421
sbatch run_pipeline.slurm
squeue -u kiana.shen22
```

Watch logs:

```bash
tail -f /scratch/user/kiana.shen22/icd_logs/<JOBID>.out
cat /scratch/user/kiana.shen22/icd_logs/<JOBID>.err
```

## 7. Run interactively for debugging

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash
cd /scratch/user/kiana.shen22/FinalProject_421
bash run_pipeline.slurm
```

## 8. Copy predictions back to Windows

From PowerShell or CMD on your computer:

```bash
scp kiana.shen22@grace.hprc.tamu.edu:/scratch/user/kiana.shen22/FinalProject_421/predictions/*.csv .
```

## Notes on what changed

- Removed the old `charu7465` paths.
- Uses `/scratch/user/kiana.shen22/python_libs` instead of forcing a new virtual environment.
- Uses offline HuggingFace mode because Grace compute nodes often cannot reach HuggingFace.
- Uses one local ClinicalBERT path by default, not a 3-model internet-dependent ensemble.
- Keeps prediction aggregation: if any sentence in a note is predicted useful, the full note row becomes label `1`.
