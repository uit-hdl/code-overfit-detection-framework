#!/usr/bin/env python

# this class should use simple_triton client to get embeddings from triton server
# then use umap_plot to reduce the dimensionality of the embeddings
# and save the reduced embeddings to a file

import argparse
import glob
import logging
import os
import pickle
import sys
from sys import stdout
from collections import defaultdict

import numcodecs
import zarr

import monai.transforms as mt
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.utils import CommonKeys
from torchvision import transforms
from tqdm import tqdm

from misc.global_util import ensure_dir_exists
from misc.monai_boilerplate import build_file_list
from network.inception_v4 import InceptionV4


def get_embeddings_bagging(model, dl, device):
    embedding_dict = {}
    model.eval()
    with torch.no_grad():
        for d in tqdm(dl, position=0, leave=True, desc="processing batch"):
            img_batch = d[CommonKeys.IMAGE]
            feat = model(img_batch.to(device))
            for i, (filename, label, f) in enumerate(zip(d["filename"], d[CommonKeys.LABEL], feat)):
                embedding_dict[filename] = {
                    CommonKeys.IMAGE: f.cpu().numpy(),
                    CommonKeys.LABEL: label,
                }
    return embedding_dict

def load_model(net, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings ')

    parser.add_argument('--src-dir', default=os.path.join('assets', 'krd-wbc', 'Dataset', 'image'), type=str,
                        help='path to dataset, folder of images')
    parser.add_argument('--feature-extractor', default='./model_dir/checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar', type=str,
                        help='path to feature extractor, which will extract features from tiles')
    parser.add_argument('--out-dir', default='out', type=str, help='path to save extracted embeddings')
    parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only',
                        dest='debug_mode')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor, device)

    transformations = mt.Compose(
        [
            mt.LoadImaged(CommonKeys.IMAGE, image_only=True),
            mt.EnsureChannelFirstd(CommonKeys.IMAGE),
            mt.EnsureTyped(CommonKeys.IMAGE),
            #mt.ToDeviceD(keys=[CommonKeys.IMAGE], device="cuda:0"),
            mt.ToTensord(CommonKeys.IMAGE, track_meta=False),
        ])

    data = []
    for filename in glob.glob(f"{args.src_dir}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(filename) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            data.append({CommonKeys.IMAGE: filename, "filename": filename})

    if args.debug_mode:
        limit = 30
        logging.warning(f"Debug mode - limiting data to {limit} samples")
        data = data[:limit]

    embedding_dest_path = os.path.join(args.out_dir, f"inception_{os.path.basename(args.src_dir)}_embedding.zarr")
    ensure_dir_exists(embedding_dest_path)
    if os.path.exists(embedding_dest_path):
        logging.error(f"Zarr file {os.path.join(os.getcwd(), embedding_dest_path)} exists - please remove it first")
        sys.exit(1)

    dl = DataLoader(dataset=Dataset(data, transformations), batch_size=64, num_workers=torch.cuda.device_count(), shuffle=False)
    embedding_dict = {}
    logging.info(f"Processing {args.src_dir}")
    model.to(device)
    model = model.eval()
    for d in tqdm(dl, position=0, leave=True, desc="processing batch"):
        img_batch = d[CommonKeys.IMAGE]
        feat = model(img_batch.to(device))
        for i, (filename, f) in enumerate(zip(d["filename"], feat)):
            embedding_dict[filename] = {
                CommonKeys.IMAGE: f.cpu().detach().numpy(),
            }

    root = zarr.group()
    for key, value in embedding_dict.items():
        dataset = root.create_dataset(
            key,
            data=value[CommonKeys.IMAGE],
            compressor=numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.AUTOSHUFFLE)
        )

    dir_store = zarr.DirectoryStore(embedding_dest_path)
    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(root.store, dir_store, log=stdout)
    logging.info(f"Wrote zarr embeddings to {embedding_dest_path}")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
