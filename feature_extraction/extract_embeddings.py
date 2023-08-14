import glob
import os
import sys

import numpy as np
from monai.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

sys.path.append('./')

import condssl.builder
import argparse
import pickle
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from network.inception_v4 import InceptionV4, InceptionV4_compat
from pathlib import Path
import monai.transforms as mt

if torch.cuda.is_available():
    device = 'cuda'
else:
    raise RuntimeError("cuda not found, use different comp :-(")
print(device)

def ensure_dir_exists(path):
    dest_dir = os.path.dirname(path)
    if not os.path.exists(dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)

def get_embeddings_bagging(feature_extractor, dl, do_outcomes):
    embedding_dict = defaultdict(list)
    outcomes_dict = defaultdict(list)
    feature_extractor.eval()
    #subtype_model.eval()
    with torch.no_grad():
        for d in tqdm(dl, position=0, leave=True, desc="processing batch"):
            # bag idx is same as slide_id. 0 is first slide, 1 is second, etc
            #img_batch, _, bag_idx = batch
            img_batch = d['image']
            bag_idx = d['slide_id']

            feat = feature_extractor(img_batch) 

            for f,bag in zip(feat, bag_idx):
                embedding_dict[bag].append(f[np.newaxis, :].cpu().numpy())
            #subtype_prob = subtype_model(img_batch)
            #subtype_pred = torch.argmax(subtype_prob, dim=1)
            #tumor_filter = (subtype_pred != 0).cpu().numpy()
            # Feat and bag_idx now has images with tumors, those that didn't are excluded
            #feat = feat[tumor_filter].cpu().numpy()

            # For each image in current batch that has tumors
            # for i, val in enumerate(tumor_filter):
            #     if not val:
            #         continue
            #     if do_outcomes:
            #         outcomes_dict[slide_index] = annotations[case_id]
                #embedding_dict[bag_idx[i]].append(feat[i][np.newaxis, :])
        # The next for loop is more about making tensors into numpy arrays. We prune away the first dimension which doesn't need to exist
        # it is not merging all tiles from slides
        for slide_id in embedding_dict:
            # flatten the array: np.concatenate(np.array([[1,2],[3,4]]), axis=0) = array([1, 2, 3, 4])
            embedding_dict[slide_id] = np.concatenate(embedding_dict[slide_id], axis=0)
    # Embedding dict now has all tensors for each tiles with tumour, grouped by slide
    # Outcomes_dict has annotation info for all slides with tumorous tile, e.g.
    # ... {0: {'recurrence': 0, 'slide_id': ['TCGA-   ...
    return embedding_dict, outcomes_dict

def load_model(net, model_dir):
    # original
    checkpoint = torch.load(model_dir)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)
    net.last_linear = nn.Identity() # the linear layer removes our dependency/link to the key encoder
    # i.e. we can write net(input) instead of net.encoder_q(input)

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--feature_extractor', default='./pretrained/checkpoint.pth.tar', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--subtype_model', default='./pretrained/checkpoint.pth.tar', type=str, help='path to subtype model, which will differentiate tumor and normal')
parser.add_argument('--src_dir', type=str, help='path to preprocessed slide images')
parser.add_argument('--outcomes', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='O', help='whether to consider outcomes or not', dest='do_outcomes')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted embeddings')

args = parser.parse_args()

# cptac_annotation = pickle.load(open('../CPTAC/recurrence_annotation.pkl', 'rb'))
# annotations = {**tcga_annotation, **cptac_annotation}
if args.do_outcomes:
    annotations = {**pickle.load(open('./TCGA/recurrence_annotation.pkl', 'rb'))}
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

# subtype_model = InceptionV4(num_classes=2).to('cuda')
# cancer_subtype_model_load = torch.load(args.subtype_model)
# subtype_model.load_state_dict(cancer_subtype_model_load)
# subtype_model = nn.DataParallel(subtype_model, device_ids=device_ids)

all_data = []
for directory in glob.glob(f"{args.src_dir}{os.sep}*"):
    for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(filename):
            slide_id = os.path.basename(filename.split(os.sep)[-2])
            tile_id = os.path.basename(filename.split(os.sep)[-1])
            all_data.append({"image": filename, "tile_id": tile_id, "slide_id": slide_id })

transformations = mt.Compose(
    [
        mt.LoadImaged("image", image_only=True),
        mt.EnsureChannelFirstd("image"),
        mt.ToTensord("image", track_meta=False),
    ])

train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=42)
# note that we split the train data again, not the entire dataset
train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=42)
ds_train = Dataset(train_data, transformations)
ds_val = Dataset(validation_data, transformations)
ds_test = Dataset(test_data, transformations)

dl_train = DataLoader(ds_train, batch_size=256, num_workers=torch.cuda.device_count(), shuffle=False)
dl_val = DataLoader(ds_val, batch_size=256, num_workers=torch.cuda.device_count(), shuffle=False)
dl_test = DataLoader(ds_test, batch_size=256, num_workers=torch.cuda.device_count(), shuffle=False)

model_name = condssl.builder.MoCo.__name__
data_dir_name = [s for s in args.src_dir.split(os.sep) if s][-1].replace("_", "")

for name, dl in list(zip(['train', 'val', 'test'], [dl_train, dl_val, dl_test]))[2:]:
    print(name)
    embedding_dict, outcomes_dict = get_embeddings_bagging(feature_extractor, dl, args.do_outcomes)
    embedding_dest_path = os.path.join(args.out_dir, model_name,  "embeddings", f"{name}_{data_dir_name}_embedding.pkl")
    ensure_dir_exists(embedding_dest_path)
    pickle.dump(embedding_dict, open(embedding_dest_path, 'wb'), protocol=4)
    if args.do_outcomes:
        ensure_dir_exists(os.path.join(args.out_dir, model_name, data_dir_name, "outcomes"))
        outcomes_dest_path = os.path.join(args.out_dir, model_name, "outcomes", f"{name}_{data_dir_name}_outcomes.pkl")
        pickle.dump((outcomes_dict), open(outcomes_dest_path, 'wb'), protocol=4)
    else:
        print("Not dumping outcomes: args.do_outcomes is False")
