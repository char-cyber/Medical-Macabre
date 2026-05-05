"""
Grace-ready ICD sentence classification pipeline.

Defaults are set for Kiana's TAMU Grace scratch space, but every path can be
changed from the command line.
"""

import argparse
import os
import warnings
from pathlib import Path

import pandas as pd
import torch

warnings.filterwarnings("ignore")

DEFAULT_BASE_DIR = Path("/scratch/user/kiana.shen22/FinalProject_421")
DEFAULT_MODEL = Path("/scratch/user/kiana.shen22/Bio_ClinicalBERT")


def parse_args():
    p = argparse.ArgumentParser(description="Train/predict ICD-relevance model on Grace")
    p.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--train", type=Path, default=None, help="Training CSV with text,label columns")
    p.add_argument("--test_dir", type=Path, default=None, help="Folder containing test*_text_only.csv")
    p.add_argument("--manual", type=Path, default=None, help="Optional manual labeled CSV")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--checkpoint_dir", type=Path, default=None)
    p.add_argument("--model_cache", type=Path, default=None)
    p.add_argument("--model_name_or_path", type=str, default=str(DEFAULT_MODEL),
                   help="Local ClinicalBERT folder or HF model name. Use your saved model path here to continue training.")
    p.add_argument("--no_external", action="store_true", help="Skip external/HuggingFace augmentation")
    p.add_argument("--offline", action="store_true", default=True, help="Force local-only HuggingFace loading")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = args.base_dir
    train_csv = args.train or (base_dir / "train_data-text_and_labels.csv")
    test_dir = args.test_dir or (base_dir / "test_files")
    manual_csv = args.manual or (base_dir / "manual_labeled_751.csv")
    output_dir = args.output_dir or (base_dir / "predictions")
    checkpoint_dir = args.checkpoint_dir or (base_dir / "checkpoints")
    model_cache = args.model_cache or Path("/scratch/user/kiana.shen22/hf_cache")

    for d in [output_dir, checkpoint_dir, model_cache]:
        d.mkdir(parents=True, exist_ok=True)

    # Grace compute nodes often cannot reach HuggingFace, so default to local files only.
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    os.environ.setdefault("HF_HOME", str(model_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(model_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(model_cache))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    from data_loader import load_all_data
    from preprocessor import preprocess_dataframe
    from label_engineer import engineer_labels
    from data_augmentation import load_external_data
    from trainer import train_ensemble
    from predictor import predict_test_files

    print("\n" + "=" * 60)
    print("STAGE 1: Loading data")
    print("=" * 60)
    train_df, test_dfs = load_all_data(train_csv, test_dir, manual_csv)
    print(f"  Train rows      : {len(train_df)}")
    print(f"  Test files found: {len(test_dfs)}")

    print("\n" + "=" * 60)
    print("STAGE 2: Preprocessing")
    print("=" * 60)
    print("  Skipping sentence splitting (data already sentence-level)")

    # Keep row mapping for aggregation
    train_df["source_row"] = train_df.index

    test_dfs = {
        name: df.assign(source_row=df.index)
        for name, df in test_dfs.items()
    }

    print("\n" + "=" * 60)
    print("STAGE 3: Label engineering")
    print("=" * 60)
    # train_df = engineer_labels(train_df, label_col="label")
    print("Skipping label engineering....")
    from label_engineer import balance_training_data
    train_df = balance_training_data(train_df)
    if not args.no_external:
        print("\n" + "=" * 60)
        print("STAGE 4: External data augmentation")
        print("=" * 60)
        ext_df = load_external_data(model_cache)
        if ext_df is not None and len(ext_df) > 0:
            train_df = pd.concat([train_df, ext_df], ignore_index=True)
            print(f"  Combined train size: {len(train_df)}")
        else:
            print("  No external data loaded — continuing with local data only")
    else:
        print("\nSkipping external data augmentation (--no_external).")

    print("\n" + "=" * 60)
    print("STAGES 5–6: Training")
    print("=" * 60)
    ensemble = train_ensemble(
        train_df,
        model_names=[args.model_name_or_path],
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_cache=model_cache,
    )

    print("\n" + "=" * 60)
    print("STAGE 7: Predicting")
    print("=" * 60)
    predict_test_files(test_dfs, ensemble, output_dir, device)
    print(f"\nAll predictions written to: {output_dir}")


if __name__ == "__main__":
    main()
