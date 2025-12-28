#!/usr/bin/env python
# encoding: utf-8
"""
Given two CSV files with a structure like:

filename,predicted_label,correct_label
/Data/TCGA_LUSC/tiles/TCGA-85-A4QR-01A-01-TSA/76801_15361.jpg,3,4
/Data/TCGA_LUSC/tiles/TCGA-85-6175-01A-01-TS1/13825_29185.jpg,2,4

This script will calculate the t-test for the predicted and correct labels.
"""
import argparse
import sys
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from scipy import stats as scistats
from itertools import combinations


def fmt(v: float) -> str:
    try:
        if np.isnan(v) or np.isinf(v):
            return str(v)
    except Exception:
        pass
    return f"{v:.6f}"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute independent two-sample t-test (ttest_ind) between predicted_label and correct_label."
    )
    parser.add_argument(
        "file1",
        type=str,
        help="Path to file one with accuracies",
    )
    parser.add_argument(
        "file2",
        type=str,
        help="Path to file two with accuracies",
    )
    return parser.parse_args(argv)


def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    x = df["accuracy"].to_numpy(dtype=float)
    return x


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)

    acc1 = load_xy(args.file1)
    acc2 = load_xy(args.file2)

    t_stat, p_val = scistats.ttest_rel(acc1, acc2, nan_policy="omit")

    print(
        f"File1: {args.file1} File2: {args.file2}\n"
        f"  len1 = {len(acc1)}, len2 = {len(acc2)}\n"
        f"  mean(predicted) = {fmt(np.mean(acc1))} (sd={fmt(np.std(acc1))})\n"
        f"  t = {fmt(float(t_stat))}, two-tailed p = {fmt(float(p_val))}"
    )

if __name__ == "__main__":
    sys.exit(main())

