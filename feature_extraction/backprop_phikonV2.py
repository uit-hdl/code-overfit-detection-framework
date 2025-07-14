#!/usr/bin/env python

import argparse
import random
from transformers import AutoImageProcessor, AutoModel
import glob
import logging
import os
import pickle
import sys
from sys import stdout
from collections import defaultdict

import numcodecs
import zarr
import zarr.storage

import monai.transforms as mt
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.utils import CommonKeys
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from tqdm import tqdm
import numpy as np

from misc.global_util import ensure_dir_exists
from misc.monai_boilerplate import build_file_list
from network.inception_v4 import InceptionV4
from monai.visualize.gradient_based import GuidedBackpropGrad

# ipython extract_features_phikon2.py -- --src-dir TCGA-LUSC-testfiles --out-dir out --gpu-id 2

class PhikonWrapper(torch.nn.Module):
    """
    Wrapper around PhikonV2 model to make it compatible with MONAI's gradient-based methods.
    Returns logits tensor instead of BaseModelOutputWithPooling object.
    """
    def __init__(self, phikon_model):
        super().__init__()
        self.phikon_model = phikon_model
        
    def forward(self, x, **kwargs):
        # Get the model output
        output = self.phikon_model(x)
        # Extract the last hidden state and return the CLS token representation
        # This creates a tensor that has the .max() method MONAI expects
        logits = output.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        return logits

def compute_backprop(model, img_tensor, target_class=None):
    """
    Compute a saliency map using MONAI's GuidedBackpropGrad.

    Args:
        model           : your PhikonV2 model (in eval() mode)
        img_tensor      : input image, a FloatTensor of shape (1, C, H, W), normalized
        target_class    : integer class index to compute saliency for.
                          If None, uses the model's top prediction.

    Returns:
        saliency_map    : numpy array of shape (H, W) with values in [0, 1]
    """
    model.eval()
    
    # Wrap the model to return logits tensor
    wrapped_model = PhikonWrapper(model)
    
    # Initialize GuidedBackpropGrad with wrapped model
    guided_backprop = GuidedBackpropGrad(wrapped_model)
    
    # Compute guided backpropagation
    # Let MONAI handle the class index internally - don't pass class_idx parameter
    if target_class is not None:
        gb = guided_backprop(img_tensor, index=target_class)  # Use 'index' parameter instead
    else:
        gb = guided_backprop(img_tensor)  # Let MONAI use top prediction
    
    # Process the saliency map
    gb = gb.abs().squeeze(0)    # (C, H, W)
    gb, _ = torch.max(gb, dim=0)    # (H, W) - take max across channels
    
    # Normalize to [0, 1]
    gb_min = gb.min()
    gb_max = gb.max()
    if gb_max > gb_min:
        gb = (gb - gb_min) / (gb_max - gb_min)
    else:
        gb = torch.zeros_like(gb)  # Handle case where all values are the same
    
    return gb.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings ')

    parser.add_argument('--src-dir', default=os.path.join('/data', 'TCGA_LUSC', 'tiles'), type=str, help='path to dataset, folder of images')
    parser.add_argument('--gpu-id', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--out-dir', default='out', type=str, help='path to save extracted embeddings')
    parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only',
                        dest='debug_mode')
    args = parser.parse_args()

    #device = torch.device(f"cuda:{args.gpu-id}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")

    transformations = mt.Compose(
        [
            mt.LoadImaged(CommonKeys.IMAGE, image_only=True),
            mt.EnsureChannelFirstd(CommonKeys.IMAGE),
            mt.EnsureTyped(CommonKeys.IMAGE),
            mt.ToTensord(CommonKeys.IMAGE, track_meta=False),
        ])

    data = []
    i = 0
    for filename in glob.glob(f"{args.src_dir}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(filename) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            data.append({CommonKeys.IMAGE: filename, "filename": filename})
            if i > 1:
                break
            i += 1

    if not data:
        raise ValueError(f"No data found in {args.src_dir}")

    if args.debug_mode:
        limit = 256
        logging.warning(f"Debug mode - limiting data to {limit} samples")
        # shuffle order of "data"
        random.shuffle(data)
        data = data[:limit]

    embedding_dest_path = os.path.join(args.out_dir, f"phikon_{os.path.basename(args.src_dir)}_embedding.zarr")
    ensure_dir_exists(embedding_dest_path)
    if os.path.exists(embedding_dest_path):
        logging.error(f"Zarr file {os.path.join(os.getcwd(), embedding_dest_path)} exists - please remove it first")
        sys.exit(1)

    dl = DataLoader(dataset=Dataset(data, transformations), batch_size=1, num_workers=1, shuffle=False)
    embedding_dict = {}
    logging.info(f"Processing {args.src_dir}")
    
    saliency_maps = []
    image_paths = []

    for d in tqdm(dl, position=0, leave=True, desc="processing batch"):
        img_batch = d[CommonKeys.IMAGE]
        inputs = processor(images=img_batch, return_tensors="pt")['pixel_values']
        saliency_map = compute_backprop(model, inputs)
        image_paths.append(d["filename"][0])
        saliency_maps.append(saliency_map)
        
    # After processing batches, show each original image and its saliency map side by side, with filename
    from PIL import Image

    # Collect original image paths and filenames
    img_paths = [entry["filename"] for entry in data]
    ensure_dir_exists(args.out_dir)

    for idx, (img_path, saliency_map) in enumerate(zip(image_paths, saliency_maps)):
        tcga_name = os.path.join(os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path)).replace(os.sep, "__")

        orig_img = Image.open(img_path).convert("RGB").resize((224, 224))
        orig_img.save(os.path.join(args.out_dir, f"original_phikon_{tcga_name}.png"))


        plt.imsave(os.path.join(args.out_dir, f"saliency_phikon_{tcga_name}.png"),
                   saliency_map,
                   cmap='hot',
                   format='png',
                   )

        saliency = Image.open(os.path.join(args.out_dir, f"saliency_phikon_{tcga_name}.png")).convert("RGB")
        saliency.save(os.path.join(args.out_dir, f"saliency_phikon_{tcga_name}.png"))

        Image.blend(orig_img, saliency, 0.5).convert("RGB").save(os.path.join(args.out_dir, f"overlay_phikon_{tcga_name}.png"))





if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()