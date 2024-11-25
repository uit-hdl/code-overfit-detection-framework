#!/usr/bin/env python
"""
Evaluate a model, outputting stats to a tensorboard log
"""
import argparse
import logging
import os
import sys
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
import monai.transforms as mt
from monai.data import Dataset, DataLoader
from monai.networks import eval_mode
from monai.utils import CommonKeys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from misc.global_util import build_file_list
from misc.monai_boilerplate import init_tb_writer, plot_distributions


def evaluate_model(model, dl_test, class_map, writer, device):
    predictions = []
    gts = []
    logging.info('Evaluating model')
    wrong_predictions = defaultdict(list)
    class_map_inv = {v: k for k, v in class_map.items()}
    with eval_mode(model):
        for item in tqdm(dl_test):
            y = model(item["image"].to(device))
            prob = F.softmax(y).detach().to("cpu")
            pred = torch.argmax(prob, dim=1).numpy()

            predictions += list(class_map_inv[x] for x in pred)

            gt = item[CommonKeys.LABEL].detach().cpu().numpy()
            gts += list(class_map_inv[x] for x in gt)
            for i,(p,g) in enumerate(zip(pred, gt)):
                if p != g:
                    wrong_predictions[class_map_inv[g]].append((item["filename"][i], g, p))

    i = 1
    grid_size = min(10, min(map(lambda l: len(l), wrong_predictions.values())))
    fig, axes = plt.subplots(nrows=len(wrong_predictions), ncols=grid_size, figsize=(16, 16), sharex=True, sharey=True)
    plt.setp(axes, xticks=[], yticks=[])
    fontsize = 6
    for row,w in enumerate(wrong_predictions):
        if len(wrong_predictions) == 1 or grid_size == 1:
            axes[row].set_ylabel(f"GT: {w}", fontsize=fontsize)
        else:
            axes[row][0].set_ylabel(f"GT: {w}", fontsize=fontsize)
        for col in range(grid_size):
            image_filename, g, p = wrong_predictions[w][col]
            if len(wrong_predictions) == 1 or grid_size == 1:
                axes[col].imshow(plt.imread(image_filename))
                axes[col].set_title("pred: {}".format(class_map_inv[p]), fontsize=fontsize)
            else:
                axes[row][col].imshow(plt.imread(image_filename))
                axes[row][col].set_title("pred: {}".format(class_map_inv[p]))
            i += 1
    writer.add_figure("Wrong Predictions", fig, 0)

    labels = list(class_map_inv.values())
    plot_results(gts, predictions, labels, "test_acc", writer)

    plot_distributions(dl_test.dataset, "test", class_map_inv, writer)

def plot_results(gts, predictions, labels, title, writer):
    cm = confusion_matrix(gts, predictions, labels=labels)
    correct_classifications = sum([cm[i][i] for i in range(len(labels))])
    wrong_classifications = len(gts) - correct_classifications
    total = correct_classifications + wrong_classifications
    writer.add_scalar(title, correct_classifications / total, 0)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cmd = disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])
    fig = cmd.ax_.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(20)
    writer.add_figure(f"Confusion Matrix - {title}", cmd.figure_)