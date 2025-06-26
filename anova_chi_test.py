#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the relationship between tumor stage and institution.
Performs:
  - ANOVA on encoded tumor stages across institutions
  - Chi-squared test of independence between tumor stage and institution
Usage:
  python analyze_tumor_stage.py --input your_file.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, chi2_contingency


def main():
    parser = argparse.ArgumentParser(
        description="Run ANOVA and chi-squared test of tumor_stage vs institution"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to CSV file with columns including 'institution' and 'tumor_stage'",
    )
    parser.add_argument('--patient', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='whether to look at patient level or tile level.',
                        dest='patient')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input, dtype=str)
    # Drop rows with missing values in key columns
    df = df.dropna(subset=["institution", "tumor_stage"])

    if args.patient:
        # remove filename
        df = df[["bcr_patient_barcode", "institution", "tumor_stage"]].drop_duplicates()

    # Remove 'A' and 'B' suffixes from tumor_stage
    df["tumor_stage"] = df["tumor_stage"].str.replace(r'[AB]$', '', regex=True)
    # also remove any df["tumor_stage"] which does not start with "Stage"
    df = df[df["tumor_stage"].str.startswith("Stage")]

    #df = df[~df["tumor_stage"].str.endswith("IV")]



# Print overall tumor stage percentages
    print("Overall tumor stage distribution (%):")
    overall_pct = df["tumor_stage"].value_counts(normalize=True) * 100
    for stage, pct in overall_pct.items():
        print(f"  {stage}: {pct:.2f}%")
    print()
    

    # Print tumor stage percentages by institution
    print("Tumor stage distribution by institution (%):")
    for inst, group in df.groupby("institution"):
        pct = group["tumor_stage"].value_counts(normalize=True) * 100
        print(f"- {inst}:")
        for stage, p in pct.items():
            print(f"    {stage}: {p:.2f}%")
    print()

    # Encode tumor_stage as an ordinal numeric code
    df["tumor_stage_cat"] = pd.Categorical(df["tumor_stage"], ordered=False)
    df["stage_code"] = df["tumor_stage_cat"].cat.codes

    # Prepare groups for ANOVA: each institution is a group
    groups = []
    labels = []
    for inst, group in df.groupby("institution"):
        codes = group["stage_code"].values
        if len(codes) >= 2:
            groups.append(codes)
            labels.append(inst)
    if len(groups) < 2:
        print("Not enough institutions with >=2 samples for ANOVA.", file=sys.stderr)
    else:
        stat, pval = f_oneway(*groups)
        print("ANOVA results:")
        print(f"F-statistic = {stat:.4f}, p-value = {pval:.4e}")
        print(f"Tested institutions: {labels}\n")

    # Chi-squared test of independence
    contingency = pd.crosstab(df["institution"], df["tumor_stage"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("Chi-squared test of independence:")
    print(f"  chi2 = {chi2:.4f}, p-value = {p:.4e}, dof = {dof}")
    print("\nContingency table (institutions × tumor_stage counts):")
    print(contingency)

if __name__ == "__main__":
    main()