#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Generate a csv file for each tile with patient-level labels
There will be redundancy in the csv file, as each tile will have the same patient-level label
However, this is a teeny tiny bit of computation to compute and load, and will make it easier to load the data later

Example output:
filename,bcr_patient_barcode,institution,slide_id,disease
/data/aza4423_anders/TCGA_LUSC/tiles/TCGA-85-8288-01A-01-TS1/19969_43009.jpg,TCGA-85-8288,85,TCGA-85-8288-01A-01-TS1,Primary Tumor

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
    parser.add_argument('--clinical-file', default=os.path.join("annotations", "liu-updated-clinical.csv"), type=str, help='path to clinical annotation file from TCGA')
    parser.add_argument('--out-file', default=os.path.join('out', 'tcga-tile-annotations.csv'), type=str, help='path to output file')

    args = parser.parse_args()

    if not args.clinical_file.endswith(".csv"):
        raise ValueError("Clinical file must be a csv file")

    cdf = pd.read_csv(args.clinical_file, sep=',', dtype=str)
    # header has fields:
    # id,bcr_patient_barcode,type,age_at_initial_pathologic_diagnosis,gender,race,ajcc_pathologic_tumor_stage,clinical_stage,histological_type,histological_grade,initial_pathologic_dx_year,menopause_status,birth_days_to,vital_status,tumor_status,last_contact_days_to,death_days_to,cause_of_death,new_tumor_event_type,new_tumor_event_site,new_tumor_event_site_other,new_tumor_event_dx_days_to,treatment_outcome_first_course,margin_status,residual_tumor,OS,OS.time,DSS,DSS.time,DFI,DFI.time,PFI,PFI.time,Redaction
    # example row:
    # 5803,TCGA-05-4244,LUAD,70,MALE,[Not Available],Stage IV,[Not Applicable],Lung Adenocarcinoma,[Not Available],2009,[Not Available],-25752,Alive,TUMOR FREE,0,#N/A,[Not Available],#N/A,#N/A,#N/A,#N/A,[Not Available],#N/A,#N/A,0,0,0,0,#N/A,#N/A,0,0,
    cdf.set_index('bcr_patient_barcode', inplace=True)

    all_data = []
    for directory in args.data_dir:
        for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                tcga_label = os.path.basename(os.path.dirname(filename))
                patient_id = '-'.join(tcga_label.split("-")[0:3])
                disease_label = tcga_label.split('-')[3]
                disease = "Primary Tumor" if "01" in disease_label else "Solid Tissue Normal"
                institution = tcga_label.split('-')[1]

                gender = cdf.loc[patient_id, 'gender']
                race = cdf.loc[patient_id, 'race']
                tumor_stage = cdf.loc[patient_id, 'ajcc_pathologic_tumor_stage']

                slide_id = tcga_label
                all_data.append({"filename": filename,
                                 "bcr_patient_barcode": patient_id,
                                 "institution": institution,
                                 "gender": gender,
                                 "race": race,
                                 "tumor_stage": tumor_stage,
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
    # to get top 5 TSS (bash):
 #awk -F',' 'NR>1{count[$3]++} END{for (i in count) print count[i],i}' out/tcga-tile-annotations.csv | sort -nr | head -5 | awk '{print $2}' | xargs -I{} awk -F',' '$3 == "{}"' out/tcga-tile-annotations.csv  > out/top5TSSfiles.csv ; header="$(head -n1 out/tcga-tile-annotations.csv)" sed -i "1s/^/$header\n/" out/top5TSSfiles.csv
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
