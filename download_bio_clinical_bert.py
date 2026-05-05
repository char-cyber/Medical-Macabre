"""Run this on your laptop (or any machine with internet access) to download
Bio_ClinicalBERT into a local folder, then scp/rsync that folder to Grace.

Usage:
    python download_bio_clinical_bert.py
    scp -r bio_clinical_bert/ siyabhatkal2010@grace.hprc.tamu.edu:/scratch/user/siyabhatkal2010/pipeline/cf/

This is needed because Grace has no outbound internet. The cf training
script (train.py) reads the model from MODEL_DIR=$PWD/bio_clinical_bert.
"""

from transformers import AutoTokenizer, AutoModel

MODEL = "emilyalsentzer/Bio_ClinicalBERT"
OUT = "bio_clinical_bert"

print(f"Downloading {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

print(f"Saving to ./{OUT}/ ...")
tokenizer.save_pretrained(OUT)
model.save_pretrained(OUT)

print(f"\nDone. Files in ./{OUT}/:")
import os
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f}  ({sz/1024/1024:.1f} MB)" if sz > 1024*1024 else f"  {f}  ({sz} bytes)")

print("\nNext step:")
print(f"  scp -r {OUT}/ siyabhatkal2010@grace.hprc.tamu.edu:/scratch/user/siyabhatkal2010/pipeline/cf/")
