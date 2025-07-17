#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys

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
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input, dtype=str)
    # Drop rows with missing values in key columns
    df = df.dropna(subset=["institution", "tumor_stage"])

    # TODO: randomly sample 50 000 rows with "MALE" and 50 000 rows with "FEMALE".
    n=7920

    whites = df[df["race"] == "WHITE"].sample(n=n, random_state=42)
    blacks = df[df["race"] == "BLACK OR AFRICAN AMERICAN"].sample(n=n, random_state=42)
    asians = df[df["race"] == "ASIAN"].sample(n=n, random_state=42)
    df_balanced = pd.concat([whites, blacks, asians])

    df_balanced.to_csv("balanced_dataset_race.csv", index=False)


if __name__ == "__main__":
    main()
