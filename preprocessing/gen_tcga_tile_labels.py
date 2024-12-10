#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Generate a csv file for each tile with patient-level labels
There will be redundancy in the csv file, as each tile will have the same patient-level label
However, this is a teeny tiny bit of computation to compute and load, and will make it easier to load the data later
'''

import argparse
import logging
import os
import sys
import glob

import openslide
import pandas as pd

from misc.global_util import ensure_dir_exists


def main():
    parser = argparse.ArgumentParser(description='Generate a csv file for each tile with patient-level labels')

    parser.add_argument('--clinical-annotation-file', type=str, help='path to annotation file')
    parser.add_argument('--data-dir', type=str, help='path to tiles. Should be a directories that have structure "<root_dir>/TCGA-XX-XXXX-XXA-XX-XX/<tile>.png"')
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
    # drop the index
    #all_data.reset_index(drop=True, inplace=True)
    # configure pandas to print entire dataframes without '...'
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    logging.debug(all_data.head())
    ensure_dir_exists(args.out_file)
    all_data.to_csv(args.out_file, index=False)
    logging.info("Wrote data to {}".format(args.out_file))
#     '''
#     annotations_files looks like this:
# id,bcr_patient_barcode,type,age_at_initial_pathologic_diagnosis,gender,race,ajcc_pathologic_tumor_stage,clinical_stage,histological_type,histological_grade,initial_pathologic_dx_year,menopause_status,birth_days_to,vital_status,tumor_status,last_contact_days_to,death_days_to,cause_of_death,new_tumor_event_type,new_tumor_event_site,new_tumor_event_site_other,new_tumor_event_dx_days_to,treatment_outcome_first_course,margin_status,residual_tumor,OS,OS.time,DSS,DSS.time,DFI,DFI.time,PFI,PFI.time,Redaction
# 5803,TCGA-05-4244,LUAD,70,MALE,[Not Available],Stage IV,[Not Applicable],Lung Adenocarcinoma,[Not Available],2009,[Not Available],-25752,Alive,TUMOR FREE,0,#N/A,[Not Available],#N/A,#N/A,#N/A,#N/A,[Not Available],#N/A,#N/A,0,0,0,0,#N/A,#N/A,0,0,
#     '''
#
#     all_labels = pd.DataFrame({})
#     for annotation_file in args.annotation_file:
#         annotations = pd.read_csv(annotation_file, header=0)
#         annotations = [['bcr_patient_barcode']]
#         all_labels = pd.concat([all_labels, annotations])
#
#     logging.debug(all_labels.head())
#
#     for tile in all_data:
#         patient_id = tile["bcr_patient_barcode"]
#         tile["label"] = all_labels.loc[patient_id]["ajcc_pathologic_tumor_stage"]


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
