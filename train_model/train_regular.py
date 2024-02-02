#!/usr/bin/env python
# coding: utf-8

import logging
from collections.abc import Iterable
import argparse
import matplotlib.ticker as mticker
import csv
import os.path
import pickle
import random
import sys
import tempfile
import time
import warnings

import monai.transforms as mt
import torch
from monai.metrics import ConfusionMatrixMetric
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from ignite.metrics import Accuracy, Loss
from matplotlib import pyplot as plt
from monai.data import DataLoader, Dataset, set_track_meta, create_test_image_2d
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.engines.workflow import Events
from monai.handlers import StatsHandler, from_engine, ValidationHandler, CheckpointSaver, MeanDice, \
    TensorBoardImageHandler, TensorBoardStatsHandler
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.networks.nets import densenet121
from monai.transforms import Compose, Activationsd, AsDiscreted, EnsureTyped, EnsureChannelFirstd
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

from global_util import build_file_list, ensure_dir_exists, institution_lookup

from train_util import *
from monai.utils import Range, CommonKeys
import contextlib
no_profiling = contextlib.nullcontext()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='batch size (default: 128), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--profile', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='P', help='whether to profile training or not', dest='is_profiling')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--data-dir', default='./data/', type=str,
                    help='path to source directory')
parser.add_argument('--out-dir', default='./out/models/', type=str,
                    help='path to output directory')
parser.add_argument('--file-list-path', default='./out/files.csv', type=str, help='path to list of file splits')
parser.add_argument('--slide_annotation_file', default='annotations/slide_label/gdc_sample_sheet.2023-08-14.tsv', type=str,
                    help='"Sample sheet" from TCGA, see README.md for instructions on how to get sheet')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')

def save_data_to_csv(data, filename, label):
    ensure_dir_exists(filename)
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["epoch", label])
        for e, l in enumerate(data):
            csvwriter.writerow([e,l])

def from_engine_custom(keys, device):
    """
        Encapsulate data in dimensions to satisfy MONAI API
    """
    def _wrapper(data):
        ret = [[i[k] for i in data] for k in keys]
        return list(map(lambda l: l.unsqueeze(0), ret[0])), list(map(lambda l: torch.tensor(l).unsqueeze(0).to(device), ret[1]))

    return _wrapper


def train(dl_train, dl_val, model, optimizer, max_epochs, out_path, device):
    val_postprocessing = Compose([EnsureTyped(keys=CommonKeys.PRED),
                                  AsDiscreted(keys=CommonKeys.PRED, argmax=True),
                                  ])

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=dl_val,
        network=model,
        val_handlers=[
            StatsHandler(tag_name="train_log", output_transform=lambda x: None),
            TensorBoardStatsHandler(log_dir=os.path.join(out_path, "runs"), output_transform=lambda x: x),
            CheckpointSaver(save_dir=os.path.join(out_path, "runs"), save_dict={"net": model}, save_key_metric=True),
            ],
        additional_metrics={"val_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))},
        key_val_metric={
            "val_mean_dice": MeanDice(output_transform=from_engine_custom(keys=[CommonKeys.PRED, CommonKeys.LABEL], device=device))
        },
        postprocessing=val_postprocessing,
    )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=dl_train,
        network=model,
        optimizer=optimizer,
        loss_function=torch.nn.CrossEntropyLoss(),
        inferer=SimpleInferer(),
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))},
        additional_metrics={"train_loss": Loss(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL], first=False), loss_fn=nn.NLLLoss())},
        train_handlers=[StatsHandler(tag_name="train_loss", output_transform=from_engine([CommonKeys.LOSS], first=True)),
                        TensorBoardStatsHandler(log_dir=os.path.join(out_path, "runs"), output_transform=lambda x: x),
                        ValidationHandler(1, evaluator),
                        CheckpointSaver(save_dir=out_path, save_dict={'network': model}, save_interval=1)
                        ],
    )

    iterLosses = []
    batchSizes = []
    epoch_loss_values = []
    metric_values = []
    mean_dice_values = []
    mean_val_acc = []

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
        mean_dice_values.append(engine.state.metrics["val_mean_dice"])
        mean_val_acc.append(engine.state.metrics["val_acc"])

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training(engine):
        loss = np.average([o["loss"] for o in engine.state.output])
        batch_len = len(engine.state.batch[0])

        iterLosses.append(loss)
        batchSizes.append(batch_len)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        # the overall average loss must be weighted by batch size
        overallAverageLoss = np.average(iterLosses, weights=batchSizes)
        epoch_loss_values.append(overallAverageLoss)
        train_acc = engine.state.metrics['train_acc']
        metric_values.append(train_acc)

        # clear the contents of iter_losses and batch_sizes for the next epoch
        del iterLosses[:]
        del batchSizes[:]

        # reset iteration for next epoch
        engine.state.iteration = 0

        # fetch and report the validation metrics
        logging.info(f"evaluation for epoch {engine.state.epoch}: averageLoss: {overallAverageLoss}, epoch_loss_values: {epoch_loss_values}, training accuracy: {train_acc}")

    trainer.run()

    return epoch_loss_values, metric_values, mean_dice_values, mean_val_acc


def wrap_data(train_data, val_data, test_data, slide_annotations, labels, batch_size, workers, is_profiling):
    logging.info('Creating dataset')
    grayer = transforms.Grayscale(num_output_channels=3)
    cropper = transforms.RandomResizedCrop(299, scale=(0.2, 1.), antialias=True)
    jitterer = transforms.ColorJitter(brightness=4, contrast=0.4, saturation=0.4, hue=0.01)


    def range_func(x, y):
        #return y
        return Range(x, methods="__call__")(y) if is_profiling else y
    transformations = mt.Compose(
        [
            range_func("LoadImage", mt.LoadImaged([CommonKeys.IMAGE], image_only=True)),
            range_func("EnsureChannelFirst", mt.EnsureChannelFirstd([CommonKeys.IMAGE])),
            range_func("Crop", mt.Lambdad([CommonKeys.IMAGE], cropper)),
            range_func("ColorJitter", mt.RandLambdad([CommonKeys.IMAGE], jitterer, prob=0.8)),
            range_func("Grayscale", mt.RandLambdad([CommonKeys.IMAGE], grayer, prob=0.2)),
            range_func("Flip0", mt.RandFlipd([CommonKeys.IMAGE], prob=0.5, spatial_axis=0)),
            range_func("Flip1", mt.RandFlipd([CommonKeys.IMAGE], prob=0.5, spatial_axis=1)),
            range_func("ToTensor", mt.ToTensord([CommonKeys.IMAGE], track_meta=False)),
            range_func("EnsureType", mt.EnsureTyped([CommonKeys.IMAGE, CommonKeys.LABEL], track_meta=False)),
        ]
    )

    val_transformations = mt.Compose(
        [
            mt.LoadImaged([CommonKeys.IMAGE]),
            mt.EnsureChannelFirstd([CommonKeys.IMAGE]),
            # mt.Lambdad([CommonKeys.IMAGE], cropper),
            # mt.RandLambdad([CommonKeys.IMAGE], jitterer, prob=0.8),
            # mt.RandLambdad([CommonKeys.IMAGE], grayer, prob=0.2),
            # mt.RandFlipd([CommonKeys.IMAGE], prob=0.5, spatial_axis=0),
            # mt.RandFlipd([CommonKeys.IMAGE], prob=0.5, spatial_axis=1),
            mt.ToTensord([CommonKeys.IMAGE], track_meta=False),
            mt.EnsureTyped([CommonKeys.IMAGE, CommonKeys.LABEL], track_meta=False),
        ]
    )

    assign_labels(train_data, labels, slide_annotations)
    assign_labels(val_data, labels, slide_annotations)
    assign_labels(test_data, labels, slide_annotations)

    ds_train = Dataset(train_data, transformations)
    ds_val = Dataset(val_data, val_transformations)
    ds_test = Dataset(test_data, val_transformations)

    dl_train = DataLoader(ds_train, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=workers, shuffle=True)

    logging.info("Number of images in dataset: {}".format(len(ds_train)))
    logging.info("Number of batches in DL: {}".format(len(dl_train)))
    logging.info("Dataset Created ...")
    return dl_train, dl_val, dl_test

def assign_labels(ds, labels, annotations):
    logging.info("Processing annotations")
    labels = {l: i for i, l in enumerate(labels)}
    for entry in ds:
        filename = entry[CommonKeys.IMAGE]
        slide_id = os.path.basename(os.path.dirname(filename))
        entry[CommonKeys.LABEL] = labels[annotations.loc[slide_id, "Sample Type"]]
        del entry['q']
        del entry['k']
        del entry['filename']
    logging.info("Annotations complete")

def main():
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    if not os.path.exists(args.slide_annotation_file):
        logging.error("TCGA annotation file not found: {}".format(args.tcga_annotation_file))
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    logging.info(f"root dir for MONAI is: {root_dir}")

    logging.info('Create dataset')

    slide_annotations = pd.read_csv(args.slide_annotation_file, sep='\t', header=0)
    slide_annotations["my_slide_id"] = slide_annotations["File Name"].map(lambda s: s.split(".")[0])
    slide_annotations = slide_annotations.set_index("my_slide_id")
    labels = slide_annotations["Sample Type"].unique().tolist()

    train_data, val_data, test_data = build_file_list(args.data_dir, args.file_list_path)
    train_data = train_data[:args.batch_size * 2]
    val_data = val_data[:args.batch_size * 2]
    test_data = test_data[:args.batch_size]
    dl_train, dl_val, dl_test = wrap_data(train_data, val_data, test_data, slide_annotations, labels, args.batch_size, args.workers, args.is_profiling)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    logging.info("Batch size: {}".format(args.batch_size))

    model_name = "densenet121"
    data_dir_name = list(filter(None, args.data_dir.split(os.sep)))[-1]
    out_path = os.path.join(args.out_dir, model_name, data_dir_name, 'model')

    model = None
    model_path = os.path.join(out_path, f"network_epoch={args.epochs}.pt")
    model = densenet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)
    if os.path.exists(model_path) and False:
        logging.info(f"=> loading model '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info('Model builder done, placed on cuda()')
    else:
        logging.info("=> creating model '{}'".format('x64'))
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        logging.info('Model builder done, placed on cuda()')
        epoch_loss_values, metric_values, mean_dice_values, mean_val_acc = train(dl_train, dl_val, model, optimizer, args.epochs, out_path, device)

        how_many_batches = len(train_data) // args.batch_size
        fig = plt.figure(1, (24, 6))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        ax = fig.add_subplot(2, 2, 1)
        ax.set_title("Iteration Average Loss")
        y = epoch_loss_values
        # set label of x-axis
        ax.axes.set_xlabel("Iteration")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.plot(range(len(y)), y)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax = fig.add_subplot(2, 2, 2)
        ax.set_title("Val Mean Dice")
        y = metric_values
        ax.axes.set_xlabel("Iteration")
        ax.plot(y)
        ax = fig.add_subplot(2, 2, 3)
        # plot the mean dice values
        ax.set_title("Val Mean Dice")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        x = len(mean_dice_values)
        y = mean_dice_values
        ax.axes.set_xlabel("Epoch")
        ax.axes.set_ylim(0, 1)
        ax.plot(y)
        ax = fig.add_subplot(2, 2, 4)
        # plot the mean validation accuracy
        ax.set_title("Val Accuracy")
        x = len(mean_val_acc)
        y = mean_val_acc
        ax.axes.set_ylim(0, 1)
        ax.axes.set_xlabel("Epoch")
        ax.plot(y)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        #fig.show()
        plt.savefig(os.path.join(args.out_dir, 'train_stats.png'))

    predictions = []
    gts = []
    logging.info('Evaluating model')
    with eval_mode(model):
        for item in tqdm(dl_test):
            prob = model(item["image"].to(device)).detach().to("cpu")
            pred = torch.argmax(prob, dim=1).numpy()
            predictions += list(x for x in pred)

            gt = item["label"].detach().cpu().numpy()
            gts += list(x for x in gt)

    roc = roc_curve(gts, predictions)
    plt.figure()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(roc[0], roc[1])

    labels = slide_annotations["Sample Type"].unique().tolist()
    cm = confusion_matrix(gts, predictions, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])

    plt.show()

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #get_logger("train_log")
    main()
