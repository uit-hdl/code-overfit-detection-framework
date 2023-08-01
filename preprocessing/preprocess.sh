#!/usr/bin/env bash

ipython ./preprocessing/process_tcga.py -- --cherry ../survival_data_raw/cherry_picked.csv --followup_path ../survival_data_raw/An_Integrated_TCGA_Pan-Cancer_Clinical_Data_Resource_to_drive_high_quality_survival_outcome_analytics.xlsx --clinical_table_path ../survival_data_raw/clinical_PANCAN_patient_with_followup.csv --wsi_path /terrahome/WinterSchool/ --refer_img colorstandard.png
