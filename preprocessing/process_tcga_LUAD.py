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
# sort data based on filename
data = sorted(data, key=lambda x: x['filename'])
ds = Dataset(data, transform=None)

for slide_id in ds:
    patient_relevant_id = "-".join(os.path.basename(slide_id['filename']).split("-")[0:7]).split(".")[0]
    #tile_path = os.path.join(args.out_dir, 'TCGA', 'tiles', os.path.basename(slide_id['patient']))
    tile_path = os.path.join(args.out_dir, 'tiles', patient_relevant_id)

    if not os.path.exists(tile_path):
        Path(tile_path).mkdir(parents=True, exist_ok=True)
        #wsi_to_tiles(slide_id['filename'], tile_path, args.refer_img, args.s)
        wsi_to_tiles_no_normalization(slide_id['filename'], tile_path, args.s)
