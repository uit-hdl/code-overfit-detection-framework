#!/usr/bin/env python

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
import numpy as np
from scipy import ndimage

import monai.transforms as mt
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.utils import CommonKeys
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from misc.global_util import ensure_dir_exists
from misc.monai_boilerplate import build_file_list
from network.inception_v4 import InceptionV4
from monai.visualize.gradient_based import GuidedBackpropGrad


def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    model.load_state_dict(model_state_dict)
    model = model.to(device)


class InceptionV4Wrapper(torch.nn.Module):
    """
    Wrapper around InceptionV4 model to make it compatible with MONAI's gradient-based methods.
    Ensures the model returns a proper tensor for gradient computation.
    """
    def __init__(self, inception_model):
        super().__init__()
        self.inception_model = inception_model
        
    def forward(self, x, **kwargs):
        # Get the model output (embeddings)
        output = self.inception_model(x)
        # Return the output tensor directly - it should already be in the right format
        return output


def compute_backprop(model, img_tensor, target_class=None, smooth_saliency=False, sigma=2.0):
    """
    Compute a saliency map using MONAI's GuidedBackpropGrad for InceptionV4.

    Args:
        model           : InceptionV4 model (in eval() mode)
        img_tensor      : input image, a FloatTensor of shape (1, C, H, W), normalized
        target_class    : integer class index to compute saliency for.
                          If None, uses the model's top prediction.
        smooth_saliency : whether to apply Gaussian smoothing
        sigma           : standard deviation for Gaussian smoothing

    Returns:
        saliency_map    : numpy array of shape (H, W) with values in [0, 1]
    """
    model.eval()
    
    # Wrap the model to ensure compatibility
    wrapped_model = InceptionV4Wrapper(model)
    
    # Initialize GuidedBackpropGrad with wrapped model
    guided_backprop = GuidedBackpropGrad(wrapped_model)
    
    # Compute guided backpropagation
    if target_class is not None:
        gb = guided_backprop(img_tensor, index=target_class)
    else:
        gb = guided_backprop(img_tensor)  # Let MONAI use top prediction
    
    # Process the saliency map
    gb = gb.abs().squeeze(0)    # (C, H, W)
    gb, _ = torch.max(gb, dim=0)    # (H, W) - take max across channels
    
    # Convert to numpy for processing
    gb_numpy = gb.cpu().numpy()
    
    # Apply Gaussian smoothing to reduce artifacts
    if smooth_saliency:
        gb_numpy = ndimage.gaussian_filter(gb_numpy, sigma=sigma)
    
    # Normalize to [0, 1]
    gb_min = gb_numpy.min()
    gb_max = gb_numpy.max()
    if gb_max > gb_min:
        gb_numpy = (gb_numpy - gb_min) / (gb_max - gb_min)
    else:
        gb_numpy = np.zeros_like(gb_numpy)  # Handle case where all values are the same
    
    return gb_numpy


def compute_integrated_gradients(model, img_tensor, target_class=None, steps=50, baseline=None):
    """
    Alternative saliency method: Integrated Gradients for InceptionV4.
    """
    model.eval()
    wrapped_model = InceptionV4Wrapper(model)
    
    if baseline is None:
        baseline = torch.zeros_like(img_tensor)
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1).to(img_tensor.device)
    interpolated_inputs = baseline + alphas * (img_tensor - baseline)
    
    # Compute gradients for each interpolated input
    gradients = []
    for i in range(steps):
        input_i = interpolated_inputs[i:i+1]
        input_i.requires_grad_(True)
        
        output = wrapped_model(input_i)
        if target_class is not None:
            score = output[0, target_class]
        else:
            score = output.max()
        
        grad = torch.autograd.grad(score, input_i, create_graph=False)[0]
        gradients.append(grad)
    
    # Average the gradients and multiply by input difference
    avg_gradients = torch.mean(torch.stack(gradients), dim=0)
    integrated_grads = avg_gradients * (img_tensor - baseline)
    
    # Process the result
    ig = integrated_grads.abs().squeeze(0)    # (C, H, W)
    ig, _ = torch.max(ig, dim=0)    # (H, W)
    
    # Convert to numpy and normalize
    ig_numpy = ig.cpu().numpy()
    ig_min = ig_numpy.min()
    ig_max = ig_numpy.max()
    if ig_max > ig_min:
        ig_numpy = (ig_numpy - ig_min) / (ig_max - ig_min)
    else:
        ig_numpy = np.zeros_like(ig_numpy)
    
    return ig_numpy


def main():
    parser = argparse.ArgumentParser(description='Extract saliency maps using InceptionV4')

    parser.add_argument('--src-dir', default=os.path.join('/data', 'TCGA_LUSC-tiles'), type=str, help='path to dataset, folder of images')
    parser.add_argument('--model-pth', default=os.path.join('out', 'models', 'MoCo', 'TCGA_LUSC', 'model', 'checkpoint_MoCo_TCGA_LUSC_0200_False_m128_n0_o0_K65536.pth.tar'), type=str, help='path to model checkpoint')
    parser.add_argument('--gpu-id', default=1, type=int, help='GPU id to use.')
    parser.add_argument('--out-dir', default='out', type=str, help='path to save extracted embeddings')
    parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only',
                        dest='debug_mode')
    parser.add_argument('--saliency-method', default='guided_backprop', choices=['guided_backprop', 'integrated_gradients'], 
                        help='Saliency computation method')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = InceptionV4(num_classes=128)
    load_model(model, args.model_pth, device)
    model.eval()

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
            if i > 1:  # Limit to 4 images for saliency visualization
                break
            i += 1

    if not data:
        raise ValueError(f"No data found in {args.src_dir}")

    if args.debug_mode:
        limit = 30
        logging.warning(f"Debug mode - limiting data to {limit} samples")
        data = data[:limit]

    dl = DataLoader(dataset=Dataset(data, transformations), batch_size=1, num_workers=1, shuffle=False)
    logging.info(f"Processing {args.src_dir}")
    
    saliency_maps = []
    image_paths = []

    for d in tqdm(dl, position=0, leave=True, desc="processing batch"):
        img_batch = d[CommonKeys.IMAGE].to(device)
        
        if args.saliency_method == 'guided_backprop':
            saliency_map = compute_backprop(model, img_batch, smooth_saliency=True, sigma=2.0)
        else:  # integrated_gradients
            saliency_map = compute_integrated_gradients(model, img_batch)
            
        image_paths.append(d["filename"][0])
        saliency_maps.append(saliency_map)
        
    # Visualize results
    from PIL import Image

    for idx, (img_path, saliency_map) in enumerate(zip(image_paths, saliency_maps)):
        # Load original image in RGB
        orig_img = Image.open(img_path).convert("RGB")

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(orig_img)
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(saliency_map, cmap='hot', interpolation='nearest')
        axs[1].set_title(f"Saliency Map ({args.saliency_method})")
        axs[1].axis('off')
        
        # Overlay saliency on original image
        axs[2].imshow(orig_img)
        axs[2].imshow(saliency_map, cmap='hot', alpha=0.4, interpolation='bilinear')
        axs[2].set_title("Overlay")
        axs[2].axis('off')

        plt.tight_layout()
        fig.suptitle(os.path.join(os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path)), fontsize=6, y=1.00)

        plt.show()
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()