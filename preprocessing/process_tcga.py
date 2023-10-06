import os
import argparse
import pandas as pd
from utils import wsi_to_tiles, ensure_dir_exists
from pathlib import Path
import pickle
import glob
from monai.data import Dataset

parser = argparse.ArgumentParser(description='Process TCGA')

# I removed duplicates from the clinical file in Excel
parser.add_argument('--clinical_path', default='./annotations/TCGA/clinical_tcga.tsv', type=str)
parser.add_argument('--follow_up_path', default='./annotations/TCGA/nationwidechildrens.org_clinical_follow_up_v1.0_lusc.txt', type=str)
parser.add_argument('--wsi_path', default='/terrahome/TCGA_LUSC/', type=str)
parser.add_argument('--refer_img', default='./preprocess/colorstandard.png', type=str)
parser.add_argument('--s', default=0.9, type=float, help='The proportion of tissues')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted tiles')
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
    tile_path = os.path.join(args.out_dir, 'TCGA', 'tiles', os.path.basename(slide_id['patient']))
    if not os.path.exists(tile_path):
        Path(tile_path).mkdir(parents=True, exist_ok=True)

followUpTable = pd.read_csv(args.follow_up_path, sep='\t')
followUpTable = followUpTable.loc[followUpTable['new_tumor_event_dx_indicator'].isin({'YES', 'NO'})]
followUpTable['recurrence'] = ((followUpTable['new_tumor_event_dx_indicator'] == 'YES') &
                    (followUpTable['new_tumor_event_type'] != 'New Primary Tumor'))
followUpTable = followUpTable.sort_values(['bcr_patient_barcode', 'form_completion_date']).drop_duplicates('bcr_patient_barcode', keep='last')
followUpTable = followUpTable.set_index('bcr_patient_barcode')
clinicalTable = pd.read_csv(args.clinical_path, sep='\t').set_index('case_submitter_id')
# Get annotation
annotation = {}
for d in ds:
    patient_id = d['patient']
    clinicalRow = clinicalTable.loc[patient_id].to_dict()

    recurrence_free_days = followUpTable.loc[patient_id]['new_tumor_event_dx_days_to'] if patient_id in followUpTable.index else ''
    recurrence_free_days = int(recurrence_free_days) if recurrence_free_days.isnumeric() else None
    if not recurrence_free_days:
        print("no recurrence information for patient %s" % patient_id)
    annotation[patient_id] = {'recurrence': clinicalRow['days_to_recurrence'].isnumeric(),
                              'stage': clinicalRow['ajcc_pathologic_stage'],
                              'survival_days': int(clinicalRow['days_to_death']) if clinicalRow['days_to_death'].isnumeric() else None,
                              'survival': True if clinicalRow['vital_status'].lower() == 'alive' else False,
                              'recurrence_free_days': recurrence_free_days,
                              'age': int(clinicalRow['age_at_diagnosis']),
                              'gender': clinicalRow['gender'],
                              # XXX: original code uses annotation[case_id]['followup_days'] = pd.to_numeric(followupTable.last_contact_days_to, errors='coerce').loc[case_id]
                              # not sure which one would be correct
                              'followup_days': int(clinicalRow['days_to_last_follow_up'] if clinicalRow['days_to_last_follow_up'] else None),
                              'patient_id': patient_id}
out_file = os.path.join(os.path.join(args.out_dir, 'annotation', 'recurrence_annotation_tcga.pkl'))
ensure_dir_exists(out_file)
pickle.dump(annotation, open(out_file, 'wb'))
