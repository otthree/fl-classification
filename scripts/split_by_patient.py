"""Patient-wise stratified train/val split for tensor folder datasets.

Ensures that all scans from the same patient stay in the same split.
Stratification is based on each patient's most frequent label.

Usage:
    python scripts/split_by_patient.py \
        --csv csv_splits_all_mri_scan_list.csv \
        --output_dir csv_splits \
        --train_ratio 0.8
"""

import argparse
import os
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


def get_patient_label(group: pd.DataFrame) -> str:
    """Return the most frequent label for a patient (for stratification)."""
    return group["label"].mode().iloc[0]


def main():
    parser = argparse.ArgumentParser(description="Patient-wise stratified split")
    parser.add_argument("--csv", type=str, required=True, help="Path to master CSV")
    parser.add_argument("--output_dir", type=str, default="csv_splits", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Columns: {list(df.columns)}")

    # Validate required columns
    required = {"pt_index", "patient_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Get unique patients and their stratification label
    patient_labels = df.groupby("patient_id").apply(get_patient_label).reset_index()
    patient_labels.columns = ["patient_id", "stratify_label"]

    print(f"\nUnique patients: {len(patient_labels)}")
    print(f"Patient label distribution: {dict(Counter(patient_labels['stratify_label']))}")

    # Stratified split at patient level
    train_patients, val_patients = train_test_split(
        patient_labels,
        train_size=args.train_ratio,
        stratify=patient_labels["stratify_label"],
        random_state=args.seed,
    )

    train_df = df[df["patient_id"].isin(train_patients["patient_id"])]
    val_df = df[df["patient_id"].isin(val_patients["patient_id"])]

    # Verify no patient overlap
    train_pids = set(train_df["patient_id"].unique())
    val_pids = set(val_df["patient_id"].unique())
    overlap = train_pids & val_pids
    assert len(overlap) == 0, f"Patient overlap detected: {overlap}"

    # Print statistics
    print(f"\n--- Split Results ---")
    print(f"Train: {len(train_df)} scans from {len(train_pids)} patients")
    print(f"  Label distribution: {dict(Counter(train_df['label']))}")
    print(f"Val:   {len(val_df)} scans from {len(val_pids)} patients")
    print(f"  Label distribution: {dict(Counter(val_df['label']))}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"\nSaved: {train_path} ({len(train_df)} rows)")
    print(f"Saved: {val_path} ({len(val_df)} rows)")


if __name__ == "__main__":
    main()
