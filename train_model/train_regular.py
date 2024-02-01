#!/usr/bin/env python
# coding: utf-8

import logging
import argparse
import csv
import os.path
import pickle
import random
import sys
import tempfile
import time
import warnings

import monai.transforms as mt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from ignite.metrics import Accuracy
from matplotlib import pyplot as plt
from monai.data import DataLoader, Dataset, set_track_meta
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.engines.workflow import Events
from monai.handlers import StatsHandler, from_engine, ValidationHandler, CheckpointSaver
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.networks.nets import densenet121
from monai.transforms import Compose, Activationsd, AsDiscreted

from global_util import build_file_list, ensure_dir_exists, institution_lookup

from train_util import *
from monai.utils import Range
import contextlib
no_profiling = contextlib.nullcontext()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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
parser.add_argument('--tcga_annotation_file', default='./out/annotation/recurrence_annotation_tcga.pkl', type=str, help='path to TCGA annotations')

def save_data_to_csv(data, filename, label):
    ensure_dir_exists(filename)
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["epoch", label])
        for e, l in enumerate(data):
            csvwriter.writerow([e,l])



def train(dl_train, dl_val, model, optimizer, max_epochs, out_path, device):
    val_postprocessing = Compose([Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)])
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=dl_val,
        network=model,
        postprocessing=val_postprocessing,
    )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=dl_train,
        network=model,
        optimizer=optimizer,
        loss_function=torch.nn.CrossEntropyLoss(),
        inferer=SimpleInferer(),
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        train_handlers=[StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
                        ValidationHandler(1, evaluator, epoch_level = True),
                        CheckpointSaver(save_dir=out_path, save_dict={'network': model}, save_interval=1)
                        ],
    )

    iterLosses = []
    batchSizes = []
    epoch_loss_values = []
    metric_values = []

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training(engine):
        batch_loss = engine.state.output
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
        print(f"evaluation for epoch {engine.state.epoch}: averageLoss: {overallAverageLoss}, epoch_loss_values: {epoch_loss_values}, training accuracy: {train_acc}")

    trainer.run()

    return epoch_loss_values, metric_values

def wrap_data(train_data, val_data, test_data, tcga_annotation_file, labels, batch_size, workers, is_profiling):
    print('Creating dataset')
    grayer = transforms.Grayscale(num_output_channels=3)
    cropper = transforms.RandomResizedCrop(299, scale=(0.2, 1.))
    jitterer = transforms.ColorJitter(brightness=4, contrast=0.4, saturation=0.4, hue=0.01)


    def range_func(x, y):
        #return y
        return Range(x, methods="__call__")(y) if is_profiling else y
    transformations = mt.Compose(
        [
            range_func("LoadImage", mt.LoadImaged(["q", "k"], image_only=True)),
            range_func("EnsureChannelFirst", mt.EnsureChannelFirstd(["q", "k"])),
            range_func("Crop", mt.Lambdad(["q", "k"], cropper)),
            range_func("ColorJitter", mt.RandLambdad(["q", "k"], jitterer, prob=0.8)),
            range_func("Grayscale", mt.RandLambdad(["q", "k"], grayer, prob=0.2)),
            range_func("Flip0", mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=0)),
            range_func("Flip1", mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=1)),
            range_func("ToTensor", mt.ToTensord(["q", "k"], track_meta=False)),
            range_func("EnsureType", mt.EnsureTyped(["q", "k"], track_meta=False)),
        ]
    )

    val_transformations = mt.Compose(
        [
            mt.LoadImaged(["q", "k"], image_only=True),
            mt.EnsureChannelFirstd(["q", "k"]),
            mt.Lambdad(["q", "k"], cropper),
            mt.RandLambdad(["q", "k"], jitterer, prob=0.8),
            mt.RandLambdad(["q", "k"], grayer, prob=0.2),
            mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=0),
            mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=1),
            mt.ToTensord(["q", "k"], track_meta=False),
        ]
    )

    ds_train = Dataset(train_data, transformations)
    ds_val = Dataset(val_data, val_transformations)
    ds_test = Dataset(test_data, val_transformations)

    tcga_annotation = pickle.load(open(tcga_annotation_file, 'rb')) if os.path.exists(tcga_annotation_file) else {}
    assign_labels(ds_train, labels, tcga_annotation)
    assign_labels(ds_val, labels, tcga_annotation)
    assign_labels(ds_test, labels, tcga_annotation)

    dl_train = DataLoader(ds_train, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=workers, shuffle=True)

    logging.info("Number of images in dataset: {}".format(len(ds_train)))
    logging.info("Number of batches in DL: {}".format(len(dl_train)))
    logging.info("Dataset Created ...")
    return dl_train, dl_val, dl_test

def assign_labels(ds, labels, annotations):
    logging.info("Processing annotations")
    # TODO: this is slow. Can speed up the lookups?
    for i in range(len(ds)):
        filename = ds[i]['filename']
        slide_id = os.path.basename(os.path.dirname(filename))
        ann = annotations.loc[slide_id]
        ds[i]['label'] = labels.index(ann["Sample Type"])
    logging.info("Annotations complete")

def main():
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    logging.info(f"root dir for MONAI is: {root_dir}")

    logging.info('Create dataset')

    train_data, val_data, test_data = build_file_list(args.data_dir, args.file_list_path)
    labels = ["Normal", "Tumor"]
    dl_train, dl_val, dl_test = wrap_data(train_data, val_data, test_data, args.tcga_annotation_file, labels, args.batch_size, args.workers)
    logging.info("Dataset Created ...")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    logging.info("=> creating model '{}'".format('x64'))
    model = densenet121(spatial_dims=2, in_channels=1, out_channels=6).to("cuda:0")
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    logging.info('Model builder done, placed on cuda()')
    logging.info("Batch size: {}".format(args.batch_size))

    model_name = "densenet121"
    data_dir_name = list(filter(None, args.data_dir.split(os.sep)))[-1]
    out_path = os.path.join(args.out_dir, model_name, data_dir_name, 'model')

    epoch_loss_values, metric_values = train(dl_train, dl_val, model, optimizer, args.epochs, out_path, device)

    max_items_to_print = 20 if len(dl_test.dataset) > 20 else len(dl_test.dataset)
    with eval_mode(model):
        for item in dl_test:
            prob = torch.squeeze(model(item["image"].to(device)).detach().to("cpu")).numpy()
            pred = labels[prob.argmax()]
            gt = labels[(item["label"].detach().item())]
            print(f"Class prediction is {pred}:{institution_lookup[pred]}. Ground-truth: {gt}:{institution_lookup[gt]}")
            max_items_to_print -= 1

    how_many_batches = len(train_data) // args.batch_size

    plt.figure(1, (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [how_many_batches * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [how_many_batches * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.grid()
    #plt.show()
    plt.savefig(os.path.join(args.out_dir, 'train_stats.png'))

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #get_logger("train_log")
    main()
