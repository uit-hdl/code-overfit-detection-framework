#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Generate a csv file for each tile with patient-level labels
There will be redundancy in the csv file, as each tile will have the same patient-level label
However, this is a teeny tiny bit of computation to compute and load, and will make it easier to load the data later
'''

import argparse
import glob
import logging
import os
import pandas as pd
import sys

from misc.global_util import ensure_dir_exists


def main():
    parser = argparse.ArgumentParser(description='Generate a csv file for each tile with patient-level labels')

    parser.add_argument('--data-dir', nargs='+', type=str, help='path to tiles. Should be a directories that have structure "<root_dir>/TCGA-XX-XXXX-XXA-XX-XX/<tile>.png"')
    parser.add_argument('--out-file', default=os.path.join('out', 'tcga-tile-annotations.csv'), type=str, help='path to output file')

    args = parser.parse_args()

    all_data = []
    for directory in args.data_dir:
        for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                tcga_label = os.path.basename(os.path.dirname(filename))
                patient_id = '-'.join(tcga_label.split("-")[0:3])
                disease_label = tcga_label.split('-')[3]
                disease = "Primary Tumor" if disease_label == "01" else "Solid Tissue Normal"
                institution = tcga_label.split('-')[1]
                slide_id = tcga_label
                all_data.append({"filename": filename,
                                 "bcr_patient_barcode": patient_id,
                                 "institution": institution,
                                 "slide_id": slide_id,
                                 "disease": disease,
                                 })
    #logging.debug(all_data[:10])
    # convert all_data to a pandas dataframe
    all_data = pd.DataFrame(all_data).reset_index(drop=True)
    # configure pandas to print entire dataframes without '...'
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    logging.debug(all_data.head())
    ensure_dir_exists(args.out_file)
    all_data.to_csv(args.out_file, index=False)
    logging.info("Wrote data to {}".format(args.out_file))

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
