import os
import argparse
import sys
import pandas as pd
from utils import wsi_to_tiles
import pickle
from collections import defaultdict


parser = argparse.ArgumentParser(description='Process TCGA')

parser.add_argument('--followup_path', default='./clinical_follow_up_v1.0_lusc.xlsx', type=str)
parser.add_argument('--clinical_table_path', default='./clinical_follow_up_v1.0_lusc.xlsx', type=str)
parser.add_argument('--cherry', default='./survival_data_raw/cherry_picked.csv', type=str)
parser.add_argument('--wsi_path', default='./TCGA_WSI', type=str)
parser.add_argument('--refer_img', default='./colorstandard.png', type=str)
parser.add_argument('--s', default=0.9, type=float, help='The proportion of tissues')


args = parser.parse_args()
extra_info_cherry = pd.read_csv(args.cherry, sep=',').set_index('bcr_patient_barcode')

# tiles_dir='/terrahome/WinterSchool/data_dir/TCGA/tiles/'
# wsi_to_tiles(0, '/terrahome/WinterSchool/eb8003bb-d5fd-4745-829c-6c5a3a43d969/TCGA-85-8287-01A-01-BS1.730a979f-7e48-4af4-8191-aa49a13054b7.svs', args.refer_img, args.s, tiles_dir)
# sys.exit(0)

followupTable = pd.read_excel(args.followup_path, skiprows=[], engine='openpyxl')
followupTable = pd.merge(followupTable, extra_info_cherry, on = 'bcr_patient_barcode', how='right')
followupTable = followupTable.loc[followupTable['new_tumor_event_dx_indicator'].isin({'YES', 'NO'})]
followupTable['recurrence'] = ((followupTable['new_tumor_event_dx_indicator'] == 'YES') &
                    (followupTable['new_tumor_event_type'] != 'New Primary Tumor'))
followupTable = followupTable.sort_values(['bcr_patient_barcode', 'form_completion_date']).drop_duplicates('bcr_patient_barcode', keep='last')
LUSC_patientids = set(followupTable['bcr_patient_barcode'])
followupTable = followupTable.set_index('bcr_patient_barcode')
hacky_cmd = ' -or '.join(list(map(lambda s: f"-name {s}\*", followupTable.index)))

tiles_dir='/terrahome/WinterSchool/data_dir/TCGA/tiles/'
wsi_list = os.popen("find {} {}".format(args.wsi_path, hacky_cmd)).read().strip('\n').split('\n')
already_processed = os.popen("find {} -maxdepth 1 -type d ".format(tiles_dir)).read().strip('\n').split('\n')
already_processed = list(map(lambda s: s.rsplit('/', 1)[1].split('.')[0], already_processed[1:]))
wsi_list_LUSC = []
for idx in range(len(wsi_list)):
    slide_id = wsi_list[idx].rsplit('/', 1)[1].split('.')[0]
    patient_id = '-'.join(slide_id.split('-', 3)[:3])
    tile_path = os.path.join(tiles_dir, slide_id)
    if patient_id in LUSC_patientids:
        if not os.path.exists(tile_path):
            os.mkdir(tile_path)
        wsi_list_LUSC.append(wsi_list[idx])

# for idx, wsi in enumerate(wsi_list_LUSC):
#     wsi_lookup = wsi.rsplit('/', 1)[1].split('.')[0]
    # if wsi_lookup in already_processed:
    #     print("Already processed {}, skipping".format(wsi_lookup))
    #     continue
    # wsi_to_tiles(idx, wsi, args.refer_img, args.s, tiles_dir)

# Get annotation
clinicalTable = followupTable #pd.read_csv(args.clinical_table_path, sep='\t').set_index('bcr_patient_barcode')
annotation = defaultdict(lambda: {"recurrence": None, "slide_id": []})
slide_ids = os.listdir(tiles_dir)
included_slides = [s for s in slide_ids if s.rsplit('-',3)[0] in set(followupTable.index)]
for slide_id in included_slides:
    case_id = '-'.join(slide_id.split('-', 3)[:3])
    clinicalRow = followupTable.loc[case_id].to_dict()
    annotation[case_id]['recurrence'] = 1 if clinicalRow['recurrence'] else 0
    annotation[case_id]['slide_id'].append(slide_id)
    annotation[case_id]['stage'] = clinicalTable.loc[case_id]['ajcc_pathologic_tumor_stage']
    annotation[case_id]['survival_days'] = clinicalTable.loc[case_id]['death_days_to']
    annotation[case_id]['survival'] = clinicalTable.loc[case_id]['vital_status']
    annotation[case_id]['recurrence_free_days'] = pd.to_numeric(followupTable.new_tumor_event_dx_days_to, errors='coerce').loc[case_id]
    annotation[case_id]['followup_days'] = pd.to_numeric(followupTable.last_contact_days_to, errors='coerce').loc[case_id]
    annotation[case_id]['gender'] = clinicalTable['gender'].loc[case_id]
pickle.dump(dict(annotation), open('./TCGA/recurrence_annotation.pkl', 'wb'))

