#!/usr/bin/env python
"""
Fine-tune a dino.py model on a dataset of white blood cell images
"""

import logging
import os
from collections import defaultdict

import monai.transforms as mt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss
from matplotlib import pyplot as plt
from monai.data import Dataset, DataLoader
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import StatsHandler, from_engine, ValidationHandler, CheckpointSaver, TensorBoardStatsHandler
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.transforms import Compose, EnsureTyped
from monai.utils import CommonKeys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from misc.global_util import ensure_dir_exists
from misc.monai_boilerplate import plot_distributions, divide_data


class FinetuneManager:
    def __init__(self, model, labels_series, writer, device, epochs,
                 out_dir="out",
                 lr=1e-3,
                 balanced=True,
                 balanced_roundup=None,
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 batch_size=64,
                 ):
        self.model = model
        self.writer = writer
        self.device = device
        self.epochs = epochs
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.lr = lr
        self.class_map, self.dl_train, self.dl_val, self.dl_test = \
            construct_datasets(labels_series, self.batch_size, balanced, balanced_roundup, train_ratio, val_ratio, test_ratio)

        if self.writer:
            self.writer.add_text("Train ratio", str(train_ratio), 0)
            self.writer.add_text("Val ratio", str(val_ratio), 0)
            self.writer.add_text("Test ratio", str(test_ratio), 0)

            class_map_inv = {v: k for k, v in self.class_map.items()}
            plot_distributions([x[CommonKeys.LABEL] for x in self.dl_train.dataset.data], "train", class_map_inv, writer)
            plot_distributions([x[CommonKeys.LABEL] for x in self.dl_val.dataset.data], "validation", class_map_inv, writer)
            plot_distributions([x[CommonKeys.LABEL] for x in self.dl_test.dataset.data], "test", class_map_inv, writer)

    def finetune_model(self):
        if self.writer:
            self.writer.add_scalar("Learning Rate", self.lr, 0)
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        loss = nn.CrossEntropyLoss().to(self.device)

        self.train(optimizer, loss)
        self.evaluate_model()

    def train(self, optimizer, loss):
        ensure_dir_exists(self.out_dir)

        val_handlers = [
            CheckpointSaver(save_dir=self.out_dir, save_dict={"net": self.model}, epoch_level=True, save_interval=2),
        ]
        if self.writer:
            val_handlers.append(TensorBoardStatsHandler(self.writer, output_transform=lambda x: x))

        evaluator = SupervisedEvaluator(
            device=self.device,
            val_data_loader=self.dl_val,
            network=self.model,
            val_handlers=val_handlers,
            key_val_metric={
                "val_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))
            },
            postprocessing=Compose([EnsureTyped(keys=CommonKeys.PRED)]),
        )

        train_handlers = [StatsHandler(tag_name="train_loss", output_transform=from_engine([CommonKeys.LOSS], first=True)),
                          ValidationHandler(1, evaluator),
                          ]
        if self.writer:
            train_handlers.append(TensorBoardStatsHandler(self.writer, output_transform=lambda x: x))

        trainer = SupervisedTrainer(
            device=self.device,
            max_epochs=self.epochs,
            train_data_loader=self.dl_train,
            network=self.model,
            optimizer=optimizer,
            loss_function=loss,
            inferer=SimpleInferer(),
            key_train_metric={"train_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))},
            additional_metrics={"train_loss": Loss(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL], first=False), loss_fn=loss)},
            train_handlers=train_handlers,
        )

        trainer.run()

        return
    def evaluate_model(self):
        evaluate_model(self.model, self.dl_test, self.class_map, writer=self.writer, device=self.device)

def construct_datasets(labels_series, batch_size, balanced = False, balanced_roundup= None, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15):
    class_map = {c: i for i, c in enumerate(labels_series.unique())}
    filenames = labels_series.index.tolist()
    labels = [class_map[l] for l in labels_series.tolist()]
    all_data = [{CommonKeys.IMAGE: f, CommonKeys.LABEL: l, "filename": f} for f,l in zip(filenames, labels)]

    splits = divide_data(all_data, balanced=balanced, balanced_roundup=balanced_roundup,
                         separate=True,
                         train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    train_data, val_data, test_data = splits['train'], splits['validation'], splits['test']

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

    return class_map, dl_train, dl_val, dl_test

def evaluate_model(model, dl_test, class_map, device="cpu", writer=None):
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
    if wrong_predictions:
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
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No wrong predictions", fontsize=12, ha='center')

    if writer:
        writer.add_figure("Wrong Predictions", fig, 0)

    labels = list(class_map_inv.values())
    plot_results(gts, predictions, labels, "test_acc")

    plot_distributions([x[CommonKeys.LABEL] for x in dl_test.dataset.data], "test", class_map_inv, writer)

def plot_results(gts, predictions, labels, title, writer=None):
    cm = confusion_matrix(gts, predictions, labels=labels)
    correct_classifications = sum([cm[i][i] for i in range(len(labels))])
    wrong_classifications = len(gts) - correct_classifications
    total = correct_classifications + wrong_classifications
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cmd = disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])
    fig = cmd.ax_.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(20)

    if writer:
        writer.add_scalar(title, correct_classifications / total, 0)
        writer.add_figure(f"Confusion Matrix - {title}", cmd.figure_)