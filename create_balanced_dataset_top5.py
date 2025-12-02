#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import argparse
import sys
import math

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, chi2_contingency


def main():
    parser = argparse.ArgumentParser(
        description="output a dataset from the top 5 TSS with an equal number of samples per tumor stage per institution."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to CSV file with columns including 'institution' and 'tumor_stage'",
    )
    parser.add_argument(
        "-n",
        required=False,
        default=1,
        help="sample factor",
    )
    parser.add_argument(
        "--out-file",
        required=False,
        default="balanced_dataset_top5.csv",
        help="how many samples to include in the resulting dataset",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input, dtype=str)
    # Drop rows with missing values in key columns
    df = df.dropna(subset=["institution", "tumor_stage"])

    # Remove 'A' and 'B' suffixes from tumor_stage
    df["tumor_stage"] = df["tumor_stage"].str.replace(r'[AB]$', '', regex=True)
    # also remove any df["tumor_stage"] which does not start with "Stage"
    df = df[df["tumor_stage"].str.startswith("Stage")]

    # only consider stage I, II and III
    df = df[~df["tumor_stage"].str.endswith("IV")]

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

    # Get top 5 institutions by sample count
    inst_counts = df['institution'].value_counts()
    top_5_inst = inst_counts.head(5).index.tolist()
    df = df[df['institution'].isin(top_5_inst)]


    # Find minimum count per stage across institutions
    min_counts = {}
    for stage in ["Stage I", "Stage II", "Stage III"]:
        stage_counts = df[df["tumor_stage"] == stage].groupby("institution").size()
        min_counts[stage] = stage_counts.min()

    # Sample equal number of samples per stage for each institution
    balanced_dfs = []
    for inst in top_5_inst:
        inst_samples = []
        inst_data = df[df["institution"] == inst]
        for stage in ["Stage I", "Stage II", "Stage III"]:
            stage_data = inst_data[inst_data["tumor_stage"] == stage]
            sampled = stage_data.sample(n=math.ceil(int(min_counts[stage])/int(args.n)), random_state=42)
            inst_samples.append(sampled)
        balanced_dfs.append(pd.concat(inst_samples))


    # Combine all balanced samples
    balanced_df = pd.concat(balanced_dfs)
    balanced_df.to_csv(args.out_file, index=False)

    # Print tumor stage percentages by institution
    #print("Tumor stage distribution by institution (%):")
    #for inst, group in balanced_df.groupby("institution"):
    #    pct = group["tumor_stage"].value_counts(normalize=True) * 100
    #    print(f"- {inst}:")
    #    for stage, p in pct.items():
    #        print(f"    {stage}: {p:.2f}%")
    #print()

    #df = balanced_df


    ## Encode tumor_stage as an ordinal numeric code
    #df["tumor_stage_cat"] = pd.Categorical(df["tumor_stage"], ordered=False)
    #df["stage_code"] = df["tumor_stage_cat"].cat.codes


    ## Prepare groups for ANOVA: each institution is a group
    #groups = []
    #labels = []
    #for inst, group in df.groupby("institution"):
    #    codes = group["stage_code"].values
    #    if len(codes) >= 2:
    #        groups.append(codes)
    #        labels.append(inst)
    #print(groups)
    #if len(groups) < 2:
    #    print("Not enough institutions with >=2 samples for ANOVA.", file=sys.stderr)
    #else:
    #    stat, pval = f_oneway(*groups)
    #    print("ANOVA results:")
    #    print(f"F-statistic = {stat:.4f}, p-value = {pval:.4e}")
    #    print(f"Tested institutions: {labels}\n")

    ## Chi-squared test of independence
    #contingency = pd.crosstab(df["institution"], df["tumor_stage"])
    #chi2, p, dof, expected = chi2_contingency(contingency)
    #print("Chi-squared test of independence:")
    #print(f"  chi2 = {chi2:.4f}, p-value = {p:.4e}, dof = {dof}")
    #print("\nContingency table (institutions × tumor_stage counts):")
    #print(contingency)


if __name__ == "__main__":
    main()
