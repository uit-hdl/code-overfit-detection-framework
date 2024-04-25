import os
import argparse
import pandas as pd
from utils import wsi_to_tiles, ensure_dir_exists
from pathlib import Path
import pickle
import glob
from monai.data import Dataset

parser = argparse.ArgumentParser(description='Process CTPAC')

parser.add_argument('--followup_path', default='./annotations/CPTAC/clinical.csv', type=str)
parser.add_argument('--wsi_path', default='/terrahome/CPTAC_LSCC/', type=str)
parser.add_argument('--refer_img', default='./preprocessing/colorstandard.png', type=str)
parser.add_argument('--s', default=0.9, type=float, help='The proportion of tissues')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted tiles')
args = parser.parse_args()

clinicalTable = pd.read_csv(args.followup_path, sep=';').set_index('case_submitter_id')
wsi_dir_dict = {}
def add_dir(directory):
    all_data = []
    for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(filename):
            all_data.append({"q": filename, "k": filename, 'filename': filename})
    return all_data

data = []
for i, filename in enumerate(glob.glob(f"{args.wsi_path}{os.sep}*")):
        if os.path.isfile(filename):
            data.append({'filename': filename, 'patient': '-'.join(os.path.basename(filename).split('.')[0].split("-"))})
ds = Dataset(data, transform=None)

for slide_id in ds:
    tile_path = os.path.join(args.out_dir, 'CPTAC', 'tiles', os.path.basename(slide_id['patient']))
    if not os.path.exists(tile_path):
        Path(tile_path).mkdir(parents=True, exist_ok=True)
        wsi_to_tiles(slide_id['filename'], tile_path, args.refer_img, args.s)


# Get annotation
annotation = {}
#for case_id in clinicalTable.index:
for d in ds:
    patient_id = d['patient']
    if not patient_id in clinicalTable.index:
        #print("Could not find patient id %s, skipping" % patient_id)
        continue
    else:
        print ("Found patient id %s" % patient_id)
    clinicalRow = clinicalTable.loc[patient_id].to_dict()
    did_recur = clinicalRow['days_to_recurrence'] is not None or clinicalRow['days_to_recurrence'] != "'--"
    annotation[patient_id] = {'recurrence': did_recur,
                           'stage': clinicalRow['ajcc_pathologic_stage'],
                           'survival_days': int(clinicalRow['days_to_death']) if clinicalRow['days_to_death'] != "'--" else None,
                           'survival': True if clinicalRow['vital_status'] == 'alive' else False,
                           'recurrence_free_days': int(clinicalRow['days_to_recurrence']) if did_recur and clinicalRow['days_to_recurrence'].isnumeric() else None,
                           'age': int(clinicalRow['age_at_diagnosis']) if clinicalRow['age_at_diagnosis'].isnumeric() else None,
                           'gender':clinicalRow['gender'],
                           'followup_days': int(clinicalRow['days_to_last_follow_up'].replace(".0", "")) if clinicalRow['days_to_last_follow_up'].isnumeric() else None,
                           'patient_id': patient_id}
out_file = os.path.join(os.path.join(args.out_dir, 'annotation', 'recurrence_annotation_cptac.pkl'))
ensure_dir_exists(out_file)
pickle.dump(annotation, open(out_file, 'wb'))
