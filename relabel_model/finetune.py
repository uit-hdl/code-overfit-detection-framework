#!/usr/bin/env python
"""
Fine-tune a dino.py model on a dataset of white blood cell images
"""
import argparse
import logging
import os
import sys

import monai.transforms as mt
import pandas as pd
import torch
import torch.nn as nn
from monai.data import Dataset, DataLoader
from monai.utils import CommonKeys
from torchvision import transforms

from misc.global_util import build_file_list
from misc.monai_boilerplate import train, init_tb_writer, plot_distributions
from relabel_model.evaluate_model import evaluate_model


def assess_model(model, labels, writer, out_dir="out", lr=1e-3, batch_size=64):
    """
    Assess the model on the given dataset
    """
    device = model.device

    train_data, val_data, test_data = build_file_list(os.path.join("assets", "krd-wbc-roi", "image"), labels, "train")

    transformations = mt.Compose(
        [
            mt.LoadImaged([CommonKeys.IMAGE], image_only=True),
            mt.EnsureChannelFirstd([CommonKeys.IMAGE]),
            mt.ToTensord([CommonKeys.IMAGE], track_meta=False),
            mt.EnsureTyped([CommonKeys.IMAGE, CommonKeys.LABEL], track_meta=False),
        ])

    ds_train = Dataset(train_data, transformations)
    ds_val = Dataset(val_data, transformations)
    ds_test = Dataset(test_data, transformations)

    dl_train = DataLoader(ds_train, batch_size=batch_size, drop_last=True, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    if writer:
        writer.add_scalar("Learning Rate", lr, 0)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss = nn.CrossEntropyLoss().to(device)

    train(dl_train, dl_val, model, optimizer, loss, 10, writer, device)
    class_map = {c: i for i, c in enumerate(labels.unique())}
    evaluate_model(model, dl_test, class_map, writer, device)