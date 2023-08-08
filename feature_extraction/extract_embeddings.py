import os
import sys

import numpy as np

sys.path.append('./')

import condssl.builder
import argparse
import pickle
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from dataset.dataloader import TCGA_CPTAC_Bag_Dataset
from network.inception_v4 import InceptionV4
from pathlib import Path

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)


def ensure_dir_exists(path):
    dest_dir = os.path.dirname(path)
    if not os.path.exists(dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)

def get_embeddings_bagging(feature_extractor, subtype_model, data_set):
    embedding_dict = defaultdict(list)
    outcomes_dict = defaultdict(list)
    feature_extractor.eval()
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=256, shuffle=False, num_workers=torch.cuda.device_count())
    with torch.no_grad():
        count = 1
        for batch in tqdm(data_loader, position=0, leave=True, desc="processing batch"):
            count += 1
            # bag idx is same as slide_id. 0 is first slide, 1 is second, etc
            img_batch, _, bag_idx = batch
            # feat = feature_extractor(img_batch.to(device)).cpu()

            # for each image (256 in each batch), 1536 output neurons. This is Inception-C model from original moco paper
            feat_out = feature_extractor(img_batch.to(device)) 
            subtype_model.eval()
            subtype_prob = subtype_model(img_batch)
            # For each image, binary has tumor or not
            subtype_pred = torch.argmax(subtype_prob, dim=1) # fo
            # Convert to boolean
            tumor_idx = (subtype_pred != 0).cpu().numpy()
            # Feat and bag_idx now has images with tumors, those that didn't are excluded
            feat = feat_out[tumor_idx].cpu().numpy()
            bag_idx = bag_idx[tumor_idx]
            # For each image in current batch that has tumors
            for i in range(len(bag_idx)):
                slide_index = bag_idx[i].item()
                # Append all tensor values from tile (feat[i]) into embedding_dict which is indexed by slide
                slide_id = data_set.idx2slide[slide_index]
                if "TCGA" in slide_id:
                    case_id = '-'.join(slide_id.split('-', 3)[:3])
                else:
                    case_id = slide_id.rsplit('-', 1)[0]
                outcomes_dict[slide_index] = annotations[case_id]
                embedding_dict[slide_id].append(feat[i][np.newaxis, :])
        # The next for loop is more about making tensors into numpy arrays. We prune away the first dimension which doesn't need to exist
        # it is not merging all tiles from slides
        for slide_id in embedding_dict:
            # flatten the array: np.concatenate(np.array([[1,2],[3,4]]), axis=0) = array([1, 2, 3, 4])
            embedding_dict[slide_id] = np.concatenate(embedding_dict[slide_id], axis=0)
    # Embedding dict now has all tensors for each tiles with tumour, grouped by slide
    # Outcomes_dict has annotation info for all slides with tumorous tile, e.g.
    # ... {0: {'recurrence': 0, 'slide_id': ['TCGA-   ...
    return embedding_dict, outcomes_dict

def load_pretrained(net, model_dir):

    # original
    checkpoint = torch.load(model_dir)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)
    net.last_linear = nn.Identity() # the linear layer removes our dependency/link to the key encoder
    # i.e. we can write net(input) instead of net.encoder_q(input)

    # new 

    # loading in a pretrained compatible format
    # checkpoint = torch.load(model_dir)
    # net.last_linear = nn.Identity() # the linear layer removes our dependency/link to the key encoder
    # net.load_state_dict(checkpoint)

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--feature_extractor_dir', default='./pretrained/checkpoint.pth.tar', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--subtype_model_dir', default='./subtype_cls/checkpoint.pth.tar', type=str, help='path to subtype model, which will differentiate tumor and normal')
parser.add_argument('--src_dir', type=str, help='path to preprocessed slide images')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted embeddings')

args = parser.parse_args()

tcga_annotation = pickle.load(open('./TCGA/recurrence_annotation.pkl', 'rb'))
# cptac_annotation = pickle.load(open('../CPTAC/recurrence_annotation.pkl', 'rb'))
# annotations = {**tcga_annotation, **cptac_annotation}
annotations = {**tcga_annotation}
feature_extractor = InceptionV4(num_classes=128)
load_pretrained(feature_extractor, args.feature_extractor_dir)
feature_extractor.to('cuda')
device_ids = [0]
feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids)

subtype_model = InceptionV4(num_classes=2).to('cuda')
cancer_subtype_model_load = torch.load(args.subtype_model_dir)
# FIXME: I have no idea what I'm doing?
subtype_model.last_linear = nn.Identity()
subtype_model.load_state_dict(cancer_subtype_model_load)
subtype_model = nn.DataParallel(subtype_model, device_ids=device_ids)


train_dataset = TCGA_CPTAC_Bag_Dataset(args.src_dir, 'train')
val_dataset = TCGA_CPTAC_Bag_Dataset(args.src_dir, 'val')
test_dataset = TCGA_CPTAC_Bag_Dataset(args.src_dir, 'test')

model_name = condssl.builder.MoCo.__name__
data_dir_name = args.data_dir.split(os.sep)[-1]
ensure_dir_exists(os.path.join(args.out_dir, model_name, data_dir_name, "embeddings"))
ensure_dir_exists(os.path.join(args.out_dir, model_name, data_dir_name, "outcomes"))

with torch.no_grad():
    for name, data_set in list(zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]))[2:]:
        print(name)
        embedding_dict, outcomes_dict = get_embeddings_bagging(feature_extractor, subtype_model, data_set)
        pickle.dump(embedding_dict, open(os.path.join(args.out_dir, model_name,  "embeddings", f"{name}_{data_dir_name}_embedding.pkl"), 'wb'), protocol=4)
        pickle.dump((outcomes_dict), open(os.path.join(args.out_dir, model_name, "outcomes", f"{name}_{data_dir_name}_outcomes.pkl"), 'wb'), protocol=4)

