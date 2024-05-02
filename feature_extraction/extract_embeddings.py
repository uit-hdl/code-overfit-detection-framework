import os
import sys
import logging

import numpy as np
from monai.data import DataLoader, Dataset

sys.path.append('./')
from misc.global_util import build_file_list, ensure_dir_exists

import argparse
import pickle
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from network.inception_v4 import InceptionV4
import monai.transforms as mt
from monai.utils import CommonKeys

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--feature-extractor', default='./out/', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--src-dir', default=os.path.join(os.path.abspath(os.sep), 'Data', 'TCGA_LUSC', 'tiles'), type=str, help='path to save extracted embeddings')
parser.add_argument('--out-dir', default='out', type=str, help='path to save extracted embeddings')
parser.add_argument('--file-list-path', default=os.path.join('out', 'files.csv'), type=str, help='path to list of file splits')

args = parser.parse_args()

def get_embeddings_bagging(feature_extractor, dl, device):
    embedding_dict = defaultdict(list)
    feature_extractor.eval()
    i = 0
    with torch.no_grad():
        for d in tqdm(dl, position=0, leave=True, desc="processing batch"):
            img_batch = d[CommonKeys.IMAGE]
            bag_idx = d['slide_id']
            tile_idx = d['tile_id']

            feat = feature_extractor(img_batch.to(device))

            for f,bag,tile in zip(feat, bag_idx, tile_idx):
                embedding_dict[bag].append((f[np.newaxis, :].cpu().numpy(), tile))
        # The next for loop is more about making tensors into numpy arrays. We prune away the first dimension which doesn't need to exist
        # it is not merging all tiles from slides
        for slide_id in embedding_dict:
            # flatten the array: np.concatenate(np.array([[1,2],[3,4]]), axis=0) = array([1, 2, 3, 4])
            embedding_dict[slide_id] = list(zip(np.concatenate([x[0] for x in embedding_dict[slide_id]], axis=0), [x[1] for x in embedding_dict[slide_id]]))
    # Embedding dict now has all tensors for each tiles with tumour, grouped by slide
    return embedding_dict

def load_model(net, model_path, device):
    # original
   # check if checkpoint has 'state_dict' or 'model'
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                            "encoder_q" in k}
        net.load_state_dict(model_state_dict)
        net.last_linear = nn.Identity()  # the linear layer removes our dependency/link to the key encoder
        # i.e. we can write net(input) instead of net.encoder_q(input)
    else:
        net.last_linear = nn.Sequential(
            nn.Linear(1536, 500),
        )
        del checkpoint['last_linear.2.weight']
        del checkpoint['last_linear.2.bias']
        del checkpoint['last_linear.4.weight']
        del checkpoint['last_linear.4.bias']
        net.load_state_dict(checkpoint)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("cuda not found, use different comp :-(")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_extractor = InceptionV4(num_classes=128)
    load_model(feature_extractor, args.feature_extractor, device)
    feature_extractor.to(device)
    feature_extractor = nn.DataParallel(feature_extractor)

    _train_data, _val_data, _test_data = build_file_list(args.src_dir, args.file_list_path)
    train_data, val_data, test_data = [], [], []
    for (li,dst) in [(_train_data,train_data), (_val_data,val_data), (_test_data,test_data)]:
        for entry in li:
            filename = entry['filename']
            slide_id = os.path.basename(filename.split(os.sep)[-2])
            dst.append({CommonKeys.IMAGE: filename, "tile_id": filename, "slide_id": slide_id})

    transformations = mt.Compose(
        [
            mt.LoadImaged(CommonKeys.IMAGE, image_only=True),
            mt.EnsureChannelFirstd(CommonKeys.IMAGE),
            mt.ToTensord(CommonKeys.IMAGE, track_meta=False),
            # doesnt work?
            # mt.ToDeviceD(keys="image", device=device),
        ])

    model_name = "inceptionv4"
    data_dir_name = list(filter(None, args.src_dir.split(os.sep)))[-1]

    for name, data in list(zip(['train', 'val', 'test'], [train_data, val_data, test_data])):
        logging.info(name)
        dl = DataLoader(dataset=Dataset(data, transformations), batch_size=128, num_workers=torch.cuda.device_count(), shuffle=True)
        embedding_dict = get_embeddings_bagging(feature_extractor, dl, device)
        embedding_dest_path = os.path.join(args.out_dir, model_name, os.path.basename(args.feature_extractor),  f"{name}_{data_dir_name}_embedding.pkl")
        ensure_dir_exists(embedding_dest_path)
        pickle.dump(embedding_dict, open(embedding_dest_path, 'wb'), protocol=4)
        logging.info(f"Wrote embeddings to {embedding_dest_path}")

if __name__ == "__main__":
    main()
