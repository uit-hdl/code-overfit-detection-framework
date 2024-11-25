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
import numpy as np

from misc.global_util import build_file_list, ensure_dir_exists
from misc.monai_boilerplate import train, init_tb_writer, plot_distributions
from relabel_model.evaluate_model import evaluate_model

# TODO: seems like this is mostly data preparation, should be divided into a separate function
def assess_model(model, labels_series, writer, device, out_dir="out", lr=1e-3, batch_size=64):
    """
    Assess the model on the given dataset
    """
    filenames = labels_series.index.tolist()
    class_map = {c: i for i, c in enumerate(labels_series.unique())}
    # map labels to integers
    labels = [class_map[l] for l in labels_series.tolist()]
    all_data = [{CommonKeys.IMAGE: f, CommonKeys.LABEL: l} for f,l in zip(filenames, labels)]
    train_data, val_data, test_data = np.split(all_data, [int(.8 * len(all_data)), int(.9 * len(all_data))])

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

    out_pth = os.path.join(out_dir, "relabelled")
    ensure_dir_exists(out_pth)
    train(dl_train, dl_val, model, optimizer, loss, 10, out_pth, writer, device)
    evaluate_model(model, dl_test, class_map, writer, device)