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


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute independent two-sample t-test (ttest_ind) between predicted_label and correct_label."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to CSV file(s) with columns: filename,predicted_label,correct_label",
    )
    return parser.parse_args(argv)


def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    df["correct_label"] = pd.to_numeric(df["correct_label"], errors="coerce")
    df = df.dropna(subset=["predicted_label", "correct_label"])
    x = df["predicted_label"].to_numpy(dtype=float)
    y = df["correct_label"].to_numpy(dtype=float)
    return x, y


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    exit_code = 0

    preds_by_file: dict[str, np.ndarray] = {}

    for path in args.files:
        try:
            x, y = load_xy(path)
            if x.size == 0 or y.size == 0:
                print(f"File: {path}\n  No valid rows found.", file=sys.stderr)
                exit_code = max(exit_code, 2)
                continue

            t_stat, p_val = scistats.ttest_rel(x, y, nan_policy="omit")

            n1, n2 = int(x.size), int(y.size)
            df = n1 + n2 - 2  # degrees of freedom for equal_var=True
            mean_x = float(np.mean(x)) if n1 else float("nan")
            mean_y = float(np.mean(y)) if n2 else float("nan")
            std_x = float(np.std(x, ddof=1)) if n1 > 1 else float("nan")
            std_y = float(np.std(y, ddof=1)) if n2 > 1 else float("nan")

            def fmt(v: float) -> str:
                try:
                    if np.isnan(v) or np.isinf(v):
                        return str(v)
                except Exception:
                    pass
                return f"{v:.6f}"

            print(
                f"File: {path}\n"
                f"  n_pred = {n1}, n_true = {n2}, df = {df}\n"
                f"  mean(predicted) = {fmt(mean_x)} (sd={fmt(std_x)}), mean(correct) = {fmt(mean_y)} (sd={fmt(std_y)})\n"
                f"  t = {fmt(float(t_stat))}, two-tailed p = {fmt(float(p_val))}"
            )

            # Save predictions for cross-file comparison
            preds_by_file[path] = x

        except KeyError as ke:
            print(f"Error processing '{path}': missing expected column {ke}", file=sys.stderr)
            exit_code = max(exit_code, 1)
        except Exception as e:
            print(f"Error processing '{path}': {e}", file=sys.stderr)
            exit_code = max(exit_code, 1)

    # Additional t_test_ind: compare the predictions against each other across files
    if len(preds_by_file) >= 2:
        print("Cross-file t_test_ind on predicted_label (pairwise):")
        
        output_dir = "out"
        output_file = os.path.join(output_dir, "t_test.csv")
        os.makedirs(output_dir, exist_ok=True)
        
        file_exists = os.path.isfile(output_file)
        
        with open(output_file, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("fileA,fileB,t,p,meanA,meanB\n")
                
            for (fa, xa), (fb, xb) in combinations(preds_by_file.items(), 2):
                t_stat, p_val = scistats.ttest_rel(xa, xb, nan_policy="omit")
                n1, n2 = int(xa.size), int(xb.size)
                df = n1 + n2 - 2
                mean_a, mean_b = float(np.mean(xa)), float(np.mean(xb))
                std_a = float(np.std(xa, ddof=1)) if n1 > 1 else float("nan")
                std_b = float(np.std(xb, ddof=1)) if n2 > 1 else float("nan")

                def fmt(v: float) -> str:
                    try:
                        if np.isnan(v) or np.isinf(v):
                            return str(v)
                    except Exception:
                        pass
                    return f"{v:.6f}"

                print(
                    f"  [{fa}] vs [{fb}]\n"
                    f"    nA = {n1}, nB = {n2}, df = {df}\n"
                    f"    meanA(pred) = {fmt(mean_a)} (sd={fmt(std_a)}), meanB(pred) = {fmt(mean_b)} (sd={fmt(std_b)})\n"
                    f"    t = {abs(float(t_stat)):.2f}, two-tailed p = {fmt(float(p_val))}"
                )
                
                f.write(f"{fa},{fb},{float(t_stat)},{float(p_val)},{mean_a},{mean_b}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

