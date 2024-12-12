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

    parser.add_argument('--data-dir', default=os.path.join('/data', 'CPTAC-tiles'), type=str, help='path to tiles. Should be a directories that have structure "<root_dir>/CPTAC-XXXXX-XX/<tile>.png"')
    parser.add_argument('--slide_annotation_file', default=os.path.join('annotations', 'CPTAC', 'slide.tsv'), type=str,
                        help='"Slide sheet", containing sample information, see README.md for instructions on how to get sheet')
    parser.add_argument('--sample_annotation_file', default=os.path.join('annotations', 'CPTAC', 'sample.tsv'), type=str,
                        help='"Slide sheet", containing sample information, see README.md for instructions on how to get sheet')
    parser.add_argument('--out-file', default=os.path.join('out', 'cptac-tile-annotations.csv'), type=str, help='path to output file')

    args = parser.parse_args()

    slide_annotations = pd.read_csv(args.slide_annotation_file, sep='\t', header=0)
    slide_annotations = slide_annotations[['slide_submitter_id', 'sample_id']]

    sample_annotations = pd.read_csv(args.sample_annotation_file, sep='\t', header=0)
    sample_annotations = sample_annotations[['sample_id', 'sample_type']]

    labels = slide_annotations.merge(sample_annotations, on='sample_id')
    labels = labels.set_index('slide_submitter_id')

    all_data = []
    for filename in glob.glob(f"{args.data_dir}{os.sep}**{os.sep}*", recursive=True):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            cptac_label = os.path.basename(os.path.dirname(filename))
            if cptac_label not in labels.index:
                logging.warning(f"Missing label for {filename}")
                continue
            disease = labels.loc[cptac_label]['sample_type']
            all_data.append({"filename": filename,
                             "slide_id": cptac_label,
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
