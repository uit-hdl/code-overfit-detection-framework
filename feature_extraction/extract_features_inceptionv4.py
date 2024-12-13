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


def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    model.load_state_dict(model_state_dict)
    model = model.to(device)

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings ')

    parser.add_argument('--src-dir', default=os.path.join('/data', 'TCGA_LUSC-tiles'), type=str, help='path to dataset, folder of images')
    #parser.add_argument('--model-pth', default=os.path.join('out', 'models', 'MoCo', 'TCGA_LUSC', 'model', 'checkpoint_MoCo_TCGA_LUSC_0200_False_m128_n0_o0_K128.pth.tar'), type=str, help='path to dataset, folder of images')
    #parser.add_argument('--model-pth', default=os.path.join('out', 'models', 'MoCo', 'TCGA_LUSC', 'model', 'checkpoint_MoCo_TCGA_LUSC_0200_False_m128_n0_o0_K128.pth.tar'), type=str, help='path to dataset, folder of images')
    parser.add_argument('--model-pth', default=os.path.join('out', 'models', 'MoCo', 'TCGA_LUSC', 'model', 'checkpoint_MoCo_TCGA_LUSC_0200_False_m128_n0_o0_K65536.pth.tar'), type=str, help='path to dataset, folder of images')
    parser.add_argument('--gpu-id', default=1, type=int, help='GPU id to use.')
    parser.add_argument('--out-dir', default='out', type=str, help='path to save extracted embeddings')
    parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only',
                        dest='debug_mode')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = InceptionV4(num_classes=128)
    load_model(model, args.model_pth, device)

    transformations = mt.Compose(
        [
            mt.LoadImaged(CommonKeys.IMAGE, image_only=True),
            mt.EnsureChannelFirstd(CommonKeys.IMAGE),
            mt.EnsureTyped(CommonKeys.IMAGE),
            mt.ToTensord(CommonKeys.IMAGE, track_meta=False),
            #mt.ToDeviceD(CommonKeys.IMAGE, device=device), # requires multiprocessing
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

    dl = DataLoader(dataset=Dataset(data, transformations), batch_size=16, num_workers=torch.cuda.device_count(), shuffle=False)
    embedding_dict = {}
    logging.info(f"Processing {args.src_dir}")
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
