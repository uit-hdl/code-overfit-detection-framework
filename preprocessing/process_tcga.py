import os
import argparse
import pandas as pd
from utils import wsi_to_tiles, wsi_to_tiles_no_normalization, ensure_dir_exists
from pathlib import Path
import pickle
import glob
from monai.data import Dataset

parser = argparse.ArgumentParser(description='Process TCGA')

# I removed duplicates from the clinical file in Excel
parser.add_argument('--clinical-path', default='./annotations/TCGA/clinical_tcga.tsv', type=str)
parser.add_argument('--follow-up-path', default='./annotations/TCGA/nationwidechildrens.org_clinical_follow_up_v1.0_lusc.tsv', type=str)
parser.add_argument('--wsi-path', default='/terrahome/TCGA-LUSC/', type=str)
parser.add_argument('--refer-img', default='./preprocessing/colorstandard.png', type=str)
parser.add_argument('--s', default=0.85, type=float, help='The proportion of tissues')
parser.add_argument('--out-dir', default='./out', type=str, help='path to save extracted tiles')
args = parser.parse_args()

def add_dir(directory):
    all_data = []
    for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(filename):
            all_data.append({"q": filename, "k": filename, 'filename': filename})
    return all_data

data = []
for i, directory in enumerate(glob.glob(f"{args.wsi_path}{os.sep}*")):
    for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*.svs", recursive=True):
        if os.path.isfile(filename):
            data.append({'filename': filename, 'patient': "-".join(os.path.basename(filename).split("-")[0:3])})
ds = Dataset(data, transform=None)

for slide_id in ds:
    patient_relevant_id = "-".join(os.path.basename(slide_id['filename']).split("-")[0:7]).split(".")[0]
    #tile_path = os.path.join(args.out_dir, 'TCGA', 'tiles', os.path.basename(slide_id['patient']))
    tile_path = os.path.join(args.out_dir, 'tiles', patient_relevant_id)

    if not os.path.exists(tile_path):
        Path(tile_path).mkdir(parents=True, exist_ok=True)
        wsi_to_tiles(slide_id['filename'], tile_path, args.refer_img, args.s)
        #wsi_to_tiles_no_normalization(slide_id['filename'], tile_path, args.s)

followUpTable = pd.read_csv(args.follow_up_path, sep='\t')
followUpTable = followUpTable.loc[followUpTable['new_tumor_event_dx_indicator'].isin({'YES', 'NO'})]
followUpTable['recurrence'] = ((followUpTable['new_tumor_event_dx_indicator'] == 'YES') &
                    (followUpTable['new_tumor_event_type'] != 'New Primary Tumor'))
followUpTable = followUpTable.sort_values(['bcr_patient_barcode', 'form_completion_date']).drop_duplicates('bcr_patient_barcode', keep='last')
followUpTable = followUpTable.set_index('bcr_patient_barcode')
clinicalTable = pd.read_csv(args.clinical_path, sep='\t').set_index('case_submitter_id')
new_tumor_event_types = ['Distant Metastasis', 'Distant Metastasis|New Primary Tumor', 'Locoregional Recurrence', 'Locoregional Recurrence|Distant Metastasis', 'New Primary Tumor', 'new_neoplasm_event_type']
# Get annotation
annotation = {}
for d in ds:
    patient_id = d['patient']
    clinicalRow = clinicalTable.loc[patient_id].to_dict()

    recurrence = clinicalRow['days_to_recurrence'].isnumeric() or followUpTable.loc[patient_id]['recurrence'] if patient_id in followUpTable.index else False
            
    recurrence_free_days = followUpTable.loc[patient_id]['new_tumor_event_dx_days_to'] if patient_id in followUpTable.index else ''
    recurrence_free_days = int(recurrence_free_days) if recurrence_free_days.isnumeric() else None
    # if not recurrence_free_days:
    #     print("no recurrence information for patient %s" % patient_id)
    if recurrence and not recurrence_free_days:
        print("data seems malformed with patient %s: reccurence %s, free_days %s" % (patient_id, recurrence, recurrence_free_days))

    followup_days = int(clinicalRow['days_to_last_follow_up']) if (clinicalRow['days_to_last_follow_up'] and clinicalRow['days_to_last_follow_up'].isnumeric()) else None
    annotation[patient_id] = {
                              # FIXME: always 0
                              # 'recurrence': clinicalRow['days_to_recurrence'].isnumeric(), # FIXME: always zero?
                              'recurrence': recurrence,
                              'stage': clinicalRow['ajcc_pathologic_stage'],
                              'survival_days': int(clinicalRow['days_to_death']) if clinicalRow['days_to_death'].isnumeric() else None,
                              'survival': True if clinicalRow['vital_status'].lower() == 'alive' else False,
                              'recurrence_free_days': recurrence_free_days,
                              'age': int(clinicalRow['age_at_diagnosis']) if clinicalRow['age_at_diagnosis'].isnumeric() else None,
                              'gender': clinicalRow['gender'],
                              # XXX: original code uses annotation[case_id]['followup_days'] = pd.to_numeric(followupTable.last_contact_days_to, errors='coerce').loc[case_id]
                              # not sure which one would be correct
                              'followup_days': followup_days,
                              'patient_id': patient_id}
out_file = os.path.join(os.path.join(args.out_dir, 'annotation', 'recurrence_annotation_tcga.pkl'))
ensure_dir_exists(out_file)
pickle.dump(annotation, open(out_file, 'wb'))
