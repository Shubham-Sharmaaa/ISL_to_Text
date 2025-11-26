#!/usr/bin/env python3
"""
clean_isl_csv.py

Safe CSV cleaner for isl_data.csv:
- backs up original CSV
- detects label column (string/non-numeric entries)
- moves label column to the last column if needed
- removes rows with non-numeric features (or tries to coerce)
- writes cleaned CSV (overwrites original with .cleaned suffix by default)
"""

import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np

def is_probably_label_series(s: pd.Series, threshold_non_numeric=0.1):
    # proportion of entries that are non-numeric or short alpha tokens
    N = len(s)
    if N == 0:
        return False
    non_numeric = s.apply(lambda x: pd.isna(x) or (not isinstance(x, (int, float, np.integer, np.floating)) and str(x).replace('.','',1).replace('-','',1).replace('e','',1).isdigit() == False))
    frac = non_numeric.sum() / float(N)
    # Also if many entries are alphabetic (A,B,C,SPACE,DEL), count that
    alpha_frac = s.apply(lambda x: isinstance(x, str) and x.isalpha()).sum() / float(N)
    return (frac > threshold_non_numeric) or (alpha_frac > 0.02)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/isl_data.csv", help="Path to CSV to clean")
    p.add_argument("--out", default=None, help="Output cleaned CSV path (default: same folder isl_data.cleaned.csv)")
    p.add_argument("--backup", action="store_true", help="Make a backup copy of original CSV")
    args = p.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print("ERROR: CSV not found:", csv_path)
        sys.exit(1)

    dirname = os.path.dirname(csv_path) or "."
    base = os.path.basename(csv_path)
    backup_path = os.path.join(dirname, base + ".bak")
    cleaned_path = args.out or os.path.join(dirname, base.replace(".csv", ".cleaned.csv"))

    if args.backup:
        print("Making backup:", backup_path)
        shutil.copy2(csv_path, backup_path)

    print("Loading CSV (may warn about dtypes)...")
    # Read everything as object first to inspect
    df = pd.read_csv(csv_path, dtype=object, low_memory=False)

    print("Columns:", df.columns.tolist())

    # Heuristic: find which column is the label column
    label_candidates = []
    for col in df.columns:
        s = df[col].astype(str).replace("nan", "")
        # fraction of entries that look alphabetic or short tokens
        alpha_count = s.apply(lambda x: x.isalpha()).sum()
        non_numeric_count = s.apply(lambda x: (x == "") or (not is_float_like(x))).sum()
        N = len(s)
        label_score = (alpha_count + non_numeric_count) / max(1, N)
        if label_score > 0.05:  # small threshold
            label_candidates.append((col, label_score, alpha_count, non_numeric_count))

    # fallback detection by header name
    if len(label_candidates) == 0:
        if 'label' in df.columns:
            label_col = 'label'
        else:
            # try last column
            label_col = df.columns[-1]
            print("No clear label candidate found, assuming last column:", label_col)
    else:
        label_candidates.sort(key=lambda x: x[1], reverse=True)
        label_col = label_candidates[0][0]
        print("Detected probable label column:", label_col, "score:", label_candidates[0][1])

    # Move label column to the end if it's not already
    cols = list(df.columns)
    if cols[-1] != label_col:
        cols.remove(label_col)
        cols.append(label_col)
        df = df[cols]
        print(f"Moved label column '{label_col}' to last position.")

    # Now try to coerce feature columns to numeric
    feature_cols = df.columns[:-1].tolist()
    print("Feature columns count:", len(feature_cols))

    # Coerce to numeric; invalid parsing becomes NaN
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with NaNs in features OR with label missing
    before = len(df)
    df = df.dropna(subset=feature_cols)
    df = df[df.iloc[:, -1].notna()]  # label exists
    after = len(df)
    print(f"Dropped {before - after} rows with non-numeric features or missing labels.")

    # Optionally, strip whitespace from label column
    df.iloc[:, -1] = df.iloc[:, -1].astype(str).str.strip()

    # Verify feature dimension equals 63
    if len(feature_cols) != 63:
        print("WARNING: feature columns count is", len(feature_cols), "but expected 63. This may indicate mismatched header.")
        # but still write cleaned CSV

    # Save cleaned CSV
    df.to_csv(cleaned_path, index=False)
    print("Wrote cleaned CSV:", cleaned_path)
    print("You can now re-run training using this cleaned CSV.")

def is_float_like(s):
    try:
        float(s)
        return True
    except:
        return False

if __name__ == "__main__":
    main()
