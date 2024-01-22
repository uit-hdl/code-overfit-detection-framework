#!/usr/bin/env python
# coding: utf-8

# orig paper: adds two new layers for classification and re-trains
# https://github.com/TahDeh/TCGA_Acquisition_site_project/blob/main/tss-feature-extraction.ipynb
# guide
#https://snappishproductions.com/blog/2020/05/25/image-self-supervised-training-with-pytorch-lightning.html.html
# other guide:
#https://github.com/Project-MONAI/tutorials/blob/main/modules/layer_wise_learning_rate.ipynb

import logging
import glob
import tempfile
import os
import sys

import numpy as np
from ignite.utils import convert_tensor
from matplotlib import pyplot as plt
from monai.data import DataLoader, Dataset
from monai.networks import eval_mode

from global_util import build_file_list, ensure_dir_exists

sys.path.append('./')

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
from monai.handlers import StatsHandler, from_engine, CheckpointSaver, ROCAUC, ValidationHandler
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from sklearn.mixture import GaussianMixture
from monai.inferers import SimpleInferer
from monai.optimizers import generate_param_groups
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy
from monai.apps import MedNISTDataset
from monai.apps import get_logger
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureTyped, AsDiscrete, EnsureType, Activations, Activationsd, AsDiscreted,
)

parser = argparse.ArgumentParser(description='Demographic parities')

parser.add_argument('--feature_extractor', default='./model_out2b1413ba2b3df0bcd9e2c56bdbea8d2c7f875d1e/MoCo/tiles/model/checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--tcga_annotation_file', default='./out/annotation/recurrence_annotation_tcga.pkl', type=str, help='path to TCGA annotations')
parser.add_argument('--file-list-path', default='./out/files.csv', type=str, help='path to list of file splits')
parser.add_argument('--src_dir', default='/Data/TCGA_LUSC/preprocessed/TCGA/tiles/', type=str, help='path to preprocessed slide images')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted embeddings')


def load_model(net, model_path, device):
    # original
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)
    #net.last_linear = nn.Identity() # the linear layer removes our dependency/link to the key encoder
    # i.e. we can write net(input) instead of net.encoder_q(input)

def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # note that we split the train data again, not the entire dataset
    data_dir_name = list(filter(None, args.src_dir.split(os.sep)))[-1]
    model_filename = os.path.join(args.out_dir, condssl.builder.MoCo.__name__, data_dir_name, 'model', 'relabelled_{}'.format(os.path.basename(args.feature_extractor)))

    institutions = set()

    train_data, val_data, test_data = build_file_list(args.src_dir, args.file_list_path)
    institution_map = lambda s: s['filename'].split(os.sep)[-2].split("-")[1]
    for inst in list(map(institution_map, train_data)) + list(map(institution_map, val_data)) + list(map(institution_map, test_data)):
        institutions.add(inst)
    institutions = sorted(institutions)
    # TODO: use brain
    for ds in [train_data, val_data, test_data]:
        for i in range(len(ds)):
            institution = institution_map(ds[i])
            ds[i]['label'] = institutions.index(institution)

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor, device)

    transformations = mt.Compose(
        [
            mt.LoadImaged("image", image_only=True),
            mt.EnsureChannelFirstd("image"),
            mt.ToTensord("image", track_meta=False),
            # doesnt work?
            # mt.ToDeviceD(keys="image", device=device),
        ])

    inferer = SimpleInferer()
    dl_test = DataLoader(dataset=Dataset(test_data, transformations), batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    with torch.no_grad():
        pred = inferer(inputs=dl_test, network=model)
        # prob = torch.squeeze(model(item["image"].to(device)).detach().to("cpu")).numpy()
        # pred = institutions[prob.argmax()]
        # gt = institutions[(item["label"].detach().item())]
        # print(f"Class prediction is {pred}:{institution_lookup[pred]}. Ground-truth: {gt}:{institution_lookup[gt]}")
        # max_items_to_print -= 1

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #get_logger("train_log")
    main()
