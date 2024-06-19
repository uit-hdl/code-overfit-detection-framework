#!/usr/bin/env python
# coding: utf-8
import glob

import pandas as pd

''' The header and data of the file looks like this:
sensitive_feature_0,accuracy,selection_rate,count,tp_rate,tn_rate,fp_rate,fn_rate,mean_pred
TCGA-18-3409-01A-01-BS1,1.0,1.0,54.0,1.0,0.0,0.0,0.0,1.0
'''


def main():
    fairness_dir = "/terrahome/analysis_out_4bd38f6f357adad620d3ccb9d0653aee846e60e5/"
    selection_rate = 241

    # Aggregate_scores should contain the mean of the selection_rate, accuracy, count, tp_rate, tn_rate, fp_rate, fn_rate, and mean_pred columns
    dataframes = []
    for dir in glob.glob(fairness_dir + "/*"):
        #for file in glob.glob(dir + "/*Slide_distributions_fairness.csv"):
        for file in glob.glob(dir + "/*Institution_distributions_fairness.csv"):
            df = pd.read_csv(file)
            df["sensitive_feature_0"] = df["sensitive_feature_0"].astype(str)
            dataframes.append(df)
            # Get overall accuracy
            print(df[["accuracy"]].mean().to_string())

    pass

    dataframes = []
    for dir in glob.glob(fairness_dir + "/*"):
        for file in glob.glob(dir + "/*Slide_distributions_fairness.csv"):
            df = pd.read_csv(file)
            df["sensitive_feature_0"] = df["sensitive_feature_0"].astype(str)
            dataframes.append(df)
            # Get mean accuracy for all slides with "11A" in the name
            print(df[df["sensitive_feature_0"].str.contains("11A")][["accuracy"]].mean().to_string())
            # Get mean accurcy for all slide without "11A" in the name
            print(df[~df["sensitive_feature_0"].str.contains("11A")][["accuracy"]].mean().to_string())

            # Print number of slides with selection_rate < 0.9 but bigger than 0.0
            print(df[df["selection_rate"] < 1.0][df["selection_rate"] > 0.0].shape[0])

    #combined_df = pd.concat(dataframes)
    #avg_df = combined_df.groupby("sensitive_feature_0").mean()
    #print(avg_df.to_string())
    pass

if __name__ == "__main__":
    main()