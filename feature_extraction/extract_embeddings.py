import glob
import os
import sys

import numpy as np
from monai.data import DataLoader, Dataset

sys.path.append('./')
from global_util import build_file_list, ensure_dir_exists

import condssl.builder
import argparse
import pickle
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from network.inception_v4 import InceptionV4
from pathlib import Path
import monai.transforms as mt
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--feature_extractor', default='./pretrained/checkpoint.pth.tar', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--tcga_annotation_file', default='./out/annotation/recurrence_annotation_tcga.pkl', type=str, help='path to TCGA annotations')
parser.add_argument('--cptac_annotation_file', default='./out/annotation/recurrence_annotation_cptac.pkl', type=str, help='path to CPTAC annotations')
parser.add_argument('--src_dir', default='/Data/TCGA_LUSC/preprocessed/by_class/lung_scc', type=str, help='path to preprocessed slide images')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted embeddings')
parser.add_argument('--file-list-path', default='./out/files.csv', type=str, help='path to list of file splits')

args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    raise RuntimeError("cuda not found, use different comp :-(")
print(device)

def get_embeddings_bagging(feature_extractor, dl, annotations):
    embedding_dict = defaultdict(list)
    outcomes_dict = defaultdict(list)
    feature_extractor.eval()
    #subtype_model.eval()
    i = 0
    with torch.no_grad():
        for d in tqdm(dl, position=0, leave=True, desc="processing batch"):
            img_batch = d['image']
            bag_idx = d['slide_id']
            tile_idx = d['tile_id']

            feat = feature_extractor(img_batch.to(device))

            for f,bag,tile in zip(feat, bag_idx, tile_idx):
                embedding_dict[bag].append((f[np.newaxis, :].cpu().numpy(), tile))
                slide_id = "-".join(bag.split("-")[0:3])
                outcomes_dict[slide_id] = annotations[slide_id]
        # The next for loop is more about making tensors into numpy arrays. We prune away the first dimension which doesn't need to exist
        # it is not merging all tiles from slides
        for slide_id in embedding_dict:
            # flatten the array: np.concatenate(np.array([[1,2],[3,4]]), axis=0) = array([1, 2, 3, 4])
            embedding_dict[slide_id] = list(zip(np.concatenate([x[0] for x in embedding_dict[slide_id]], axis=0), [x[1] for x in embedding_dict[slide_id]]))
    # Embedding dict now has all tensors for each tiles with tumour, grouped by slide
    # Outcomes_dict has annotation info for all slides with tumorous tile, e.g.
    # ... {0: {'recurrence': 0, 'slide_id': ['TCGA-   ...
    return embedding_dict, outcomes_dict

def load_model(net, model_path):
    # original
    checkpoint = torch.load(model_path)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)
    net.last_linear = nn.Identity() # the linear layer removes our dependency/link to the key encoder
    # i.e. we can write net(input) instead of net.encoder_q(input)

annotations = {}
tcga_annotation = pickle.load(open(args.tcga_annotation_file, 'rb')) if os.path.exists(args.tcga_annotation_file) else {}
cptac_annotation = pickle.load(open(args.cptac_annotation_file, 'rb')) if os.path.exists(args.cptac_annotation_file) else {}
# TODO: CPTAC
#annotations = {**tcga_annotation, **cptac_annotation}
annotations = {**tcga_annotation}

feature_extractor = InceptionV4(num_classes=128)
load_model(feature_extractor, args.feature_extractor)
feature_extractor.to('cuda')
device_ids = [0]
feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids)

# FIXME: broken: turns out I was supposed to use a different model, based off paper https://www.nature.com/articles/s41591-018-0177-5
# Need to replicate https://github.com/ncoudray/DeepPATH/tree/master, based on advice from
# https://github.com/NYUMedML/conditional_ssl_hist/issues/3
# ... FIXME: but that model (above) is just trained on labeled SLIDES, not tiles. So I might as well just use annotation info referenced in annotations/README.md??
# ah, but it would work on CPTAC, for which I don't know if there are annotations?

all_data = []
number_of_slides = len(glob.glob(f"{args.src_dir}{os.sep}*"))
splits = [int(number_of_slides * 0.7), int(number_of_slides * 0.1), int(number_of_slides * 0.2)]
def add_dir(directory):
    all_data = []
    for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(filename):
            slide_id = os.path.basename(filename.split(os.sep)[-2])
            tile_id = os.path.basename(filename.split(os.sep)[-1])
            all_data.append({"image": filename, "tile_id": filename, "slide_id": slide_id})
    return all_data

_train_data, _val_data, _test_data = build_file_list(args.src_dir, args.file_list_path)
train_data, val_data, test_data = [], [], []
for (li,dst) in [(_train_data,train_data), (_val_data,val_data), (_test_data,test_data)]:
    for entry in li:
        filename = entry['filename']
        slide_id = os.path.basename(filename.split(os.sep)[-2])
        tile_id = os.path.basename(filename.split(os.sep)[-1])
        dst.append({"image": filename, "tile_id": filename, "slide_id": slide_id})

transformations = mt.Compose(
    [
        mt.LoadImaged("image", image_only=True),
        mt.EnsureChannelFirstd("image"),
        mt.ToTensord("image", track_meta=False),
        # doesnt work?
        # mt.ToDeviceD(keys="image", device=device),
    ])

model_name = condssl.builder.MoCo.__name__
data_dir_name = list(filter(None, args.src_dir.split(os.sep)))[-1]

for name, data in list(zip(['train', 'val', 'test'], [train_data, val_data, test_data]))[0:]:
    print(name)
    dl = DataLoader(dataset=Dataset(data, transformations), batch_size=256, num_workers=torch.cuda.device_count(), shuffle=True)
    embedding_dict, outcomes_dict = get_embeddings_bagging(feature_extractor, dl, annotations)
    embedding_dest_path = os.path.join(args.out_dir, model_name,  data_dir_name, "embeddings", f"{name}_{data_dir_name}_embedding.pkl")
    ensure_dir_exists(embedding_dest_path)
    pickle.dump(embedding_dict, open(embedding_dest_path, 'wb'), protocol=4)
    print(f"Wrote embeddings to {embedding_dest_path}")

    outcomes_dest_path = os.path.join(args.out_dir, model_name, data_dir_name, "embeddings", f"{name}_{data_dir_name}_outcomes.pkl")
    pickle.dump((outcomes_dict), open(outcomes_dest_path, 'wb'), protocol=4)
    print(f"Wrote outcomes to {outcomes_dest_path}")
