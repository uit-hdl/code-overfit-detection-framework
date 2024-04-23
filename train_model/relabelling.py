#!/usr/bin/env python
# coding: utf-8

# orig paper: adds two new layers for classification and re-trains
# https://github.com/TahDeh/TCGA_Acquisition_site_project/blob/main/tss-feature-extraction.ipynb
# guide
#https://snappishproductions.com/blog/2020/05/25/image-self-supervised-training-with-pytorch-lightning.html.html
# other guide:
#https://github.com/Project-MONAI/tutorials/blob/main/modules/layer_wise_learning_rate.ipynb


import logging
import os
import sys
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from monai.data import DataLoader, Dataset
from monai.networks import eval_mode
from monai.networks.utils import freeze_layers

sys.path.append('./')
from misc.global_util import build_file_list, ensure_dir_exists


import pandas as pd
import condssl.builder
import argparse
import matplotlib.ticker as mticker
from monai.utils import Range, CommonKeys
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from tqdm import tqdm
from network.inception_v4 import InceptionV4
import monai.transforms as mt
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.inferers import SimpleInferer
from monai.optimizers import generate_param_groups
from ignite.engine import Events
from monai.transforms import Compose, EnsureTyped
from monai.handlers import StatsHandler, from_engine, ValidationHandler, CheckpointSaver, TensorBoardStatsHandler
from monai.handlers.tensorboard_handlers import SummaryWriter
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from ignite.metrics import Accuracy, Loss
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--feature_extractor',
                    default='./model_out2b1413ba2b3df0bcd9e2c56bdbea8d2c7f875d1e/MoCo/tiles/model/checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar',
                    type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--tcga_annotation_file', default=os.path.join('out', 'annotation', 'recurrence_annotation_tcga.pkl'), type=str, help='path to TCGA annotations')
parser.add_argument('--file-list-path', default=os.path.join('out', 'files.csv'), type=str, help='path to list of file splits')
parser.add_argument('--label-key', default='Sample Type', type=str, help='default key to use for doing fine-tuning. If set to "my_inst", will retrain using institution as label')
parser.add_argument('--src-dir', default=os.path.join('Data', 'TCGA_LUSC', 'preprocessed', 'TCGA', 'tiles'), type=str, help='path to preprocessed slide images')
parser.add_argument('--out-dir', default='./out', type=str, help='path to save extracted embeddings')
parser.add_argument('--slide_annotation_file', default=os.path.join('annotations', 'slide_label', 'gdc_sample_sheet.2023-08-14.tsv'), type=str,
                    help='"Sample sheet" from TCGA, see README.md for instructions on how to get sheet')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--profile', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='P', help='whether to profile training or not', dest='is_profiling')
parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only', dest='debug_mode')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--lr', '--learning-rate', default=1e-05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

def from_engine_custom(keys, device):
    """
        Encapsulate data in dimensions to satisfy MONAI API
    """
    def _wrapper(data):
        ret = [[i[k] for i in data] for k in keys]
        return list(map(lambda l: l.unsqueeze(0), ret[0])), list(map(lambda l: torch.tensor(l).unsqueeze(0).to(device), ret[1]))

    return _wrapper

def load_model(net, model_path, device):
    # original
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)
    #net.last_linear = nn.Identity() # the linear layer removes our dependency/link to the key encoder
    # i.e. we can write net(input) instead of net.encoder_q(input)

def attach_layers(model, num_neurons_in_layers, out_classes):
    model.last_linear = nn.Sequential(
        nn.Linear(1536, 500),
        nn.ReLU(),
        nn.Linear(500, 200),
        nn.ReLU(),
        nn.Linear(200, out_classes),
    )
    return model

def assign_labels(ds, labels, annotations, label_key):
    labels = {l: i for i, l in enumerate(labels)}
    for entry in ds:
        filename = entry[CommonKeys.IMAGE]
        slide_id = os.path.basename(os.path.dirname(filename))
        entry[CommonKeys.LABEL] = labels[annotations.loc[slide_id, label_key]]
        del entry['q']
        del entry['k']
        #del entry['filename']

def train(dl_train, dl_val, model, optimizer, max_epochs, ratio_of_positives, out_path, writer, device):
    val_postprocessing = Compose([EnsureTyped(keys=CommonKeys.PRED),
                                  # AsDiscreted(keys=CommonKeys.PRED, argmax=True),
                                  ])

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=dl_val,
        network=model,
        val_handlers=[
            StatsHandler(tag_name="train_log", output_transform=lambda x: None),
            TensorBoardStatsHandler(writer, output_transform=lambda x: x),
            CheckpointSaver(save_dir=os.path.join(out_path, "runs"), save_dict={"net": model}, save_key_metric=True),
            ],
        key_val_metric={
            "val_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))
        },
        postprocessing=val_postprocessing,
    )

    if ratio_of_positives:
        w = torch.Tensor([1 - ratio_of_positives, ratio_of_positives])
    else:
        w = None
    loss = nn.CrossEntropyLoss(w).cuda()

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=dl_train,
        network=model,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))},
        additional_metrics={"train_loss": Loss(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL], first=False), loss_fn=loss)},
        train_handlers=[StatsHandler(tag_name="train_loss", output_transform=from_engine([CommonKeys.LOSS], first=True)),
                        TensorBoardStatsHandler(writer, output_transform=lambda x: x),
                        ValidationHandler(1, evaluator),
                        CheckpointSaver(save_dir=out_path, save_dict={'network': model}, save_interval=1)
                        ],
    )

    iterLosses = []
    batchSizes = []
    epoch_loss_values = []
    metric_values = []
    mean_val_acc = []

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
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

    return epoch_loss_values, metric_values, mean_val_acc

def wrap_data(train_data, val_data, test_data, slide_annotations, labels, label_key, batch_size, workers, is_profiling):
    logging.info('Creating dataset')
    grayer = transforms.Grayscale(num_output_channels=3)
    cropper = transforms.RandomResizedCrop(299, scale=(0.2, 1.), antialias=True)
    jitterer = transforms.ColorJitter(brightness=4, contrast=0.4, saturation=0.4, hue=0.01)

    # For profiling purposes, if profiling
    def range_func(x, y):
        return Range(x, methods="__call__")(y) if is_profiling else y

    if label_key.lower() == "my_inst":
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
    else:
        transformations = mt.Compose(
            [
                mt.LoadImaged("image", image_only=True),
                mt.EnsureChannelFirstd("image"),
                mt.ToTensord("image", track_meta=False),
                # doesnt work?
                # mt.ToDeviceD(keys="image", device=device),
            ])

    val_transformations = mt.Compose(
        [
            mt.LoadImaged([CommonKeys.IMAGE]),
            mt.EnsureChannelFirstd([CommonKeys.IMAGE]),
            mt.ToTensord([CommonKeys.IMAGE], track_meta=False),
            mt.EnsureTyped([CommonKeys.IMAGE, CommonKeys.LABEL], track_meta=False),
        ]
    )

    assign_labels(train_data, labels, slide_annotations, label_key)
    assign_labels(val_data, labels, slide_annotations, label_key)
    assign_labels(test_data, labels, slide_annotations, label_key)

    ds_train = Dataset(train_data, transformations)
    ds_val = Dataset(val_data, val_transformations)
    ds_test = Dataset(test_data, val_transformations)

    dl_train = DataLoader(ds_train, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=workers, shuffle=True)

    logging.info("Dataset Created ...")
    logging.info("Number of images in train dataset: {}".format(len(ds_train)))
    logging.info("Number of batches in train DL: {}".format(len(dl_train)))
    return dl_train, dl_val, dl_test

def plot_train_data(epoch_loss_values, metric_values, mean_val_acc, out_dir):
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
    # fig.show()
    plt.savefig(os.path.join(out_dir, 'train_stats.png'))


def main():
    args = parser.parse_args()

    data_dir_name = list(filter(None, args.src_dir.split(os.sep)))[-1]
    out_path = os.path.join(args.out_dir, condssl.builder.MoCo.__name__, data_dir_name, 'model', 'relabelled_{}_{}'.format(args.label_key, os.path.basename(args.feature_extractor)))

    logfile_path = os.path.join(out_path, "output.log")
    ensure_dir_exists(logfile_path)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile_path),
            logging.StreamHandler()
        ]
    )
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    if not os.path.exists(args.slide_annotation_file):
        logging.error("TCGA annotation file not found: {}".format(args.tcga_annotation_file))
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info('Process annotations')
    slide_annotations = pd.read_csv(args.slide_annotation_file, sep='\t', header=0)
    slide_annotations["my_slide_id"] = slide_annotations["File Name"].map(lambda s: s.split(".")[0])
    slide_annotations = slide_annotations.set_index("my_slide_id")
    slide_annotations["my_inst"] = slide_annotations["File Name"].map(lambda s: s.split("-")[1])
    labels = slide_annotations[args.label_key].unique().tolist()
    labels.sort()
    assert args.label_key != 'Sample Type' or labels[0] == "Solid Tissue Normal" # making 0 the benign class if using 'sample type'

    logging.info('Creating dataset')
    train_data, val_data, test_data = build_file_list(args.src_dir, args.file_list_path)
    if args.debug_mode:
        logging.warning("Debug mode enabled!")
        np.random.shuffle(train_data)
        np.random.shuffle(val_data)
        np.random.shuffle(test_data)
        train_data = train_data[:args.batch_size * 4]
        val_data = val_data[:args.batch_size * 4]
        test_data = test_data[:args.batch_size * 4]

    dl_train, dl_val, dl_test = wrap_data(train_data, val_data, test_data, slide_annotations, labels, args.label_key, args.batch_size, args.workers, args.is_profiling)

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor, device)
    model = attach_layers(model, [500, 200], len(labels))
    freeze_layers(model, exclude_vars="last_linear")
    model.to(device)

    model_path = os.path.join(out_path, f"network_epoch={args.epochs}.pt")
    writer = SummaryWriter(log_dir=os.path.join(out_path, "runs"))
    if os.path.exists(model_path):
        logging.info(f"=> loading model '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info('Model builder done')
    else:
        logging.info("=> creating model '{}'".format('x64'))
        #model = densenet121(spatial_dims=2, in_channels=3, out_channels=len(labels), pretrained=True)
        #model.to(device)
        params = generate_param_groups(network=model, layer_matches=[lambda x: x.last_linear], match_types=["select"],
                                       lr_values=[args.lr])
        optimizer = torch.optim.Adam(params, args.lr)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logging.info('Model builder done')
        if args.label_key == 'Sample Type':
            ratio_of_positives = 1 / (len(train_data) / len(list(filter(lambda l: l["label"] == 1, train_data))))
            if not args.debug_mode and (ratio_of_positives > 0.9 or ratio_of_positives < 0.1):
                logging.error(f"Ratio of positives ({ratio_of_positives}) is very high/low, consider adjusting the dataset. Exiting in case this was a mistake :)")
                sys.exit(1)
            elif args.debug_mode:
                ratio_of_positives = 0.5
                logging.warning("Debug mode enabled, setting ratio_of_positives to 0.5")
            writer.add_scalar("ratio_of_positives_train", ratio_of_positives, 0)
        else:
            ratio_of_positives = None
        epoch_loss_values, metric_values, mean_val_acc = train(dl_train, dl_val, model, optimizer, args.epochs, ratio_of_positives, out_path, writer, device)
        writer.add_scalar("size of dataset", len(train_data), 0)
        writer.add_scalar("batch size", args.batch_size, 0)
        writer.add_text("model name", "inception")
        writer.add_text("model_path", model_path)
        writer.add_text("data dir", data_dir_name)
        writer.add_text("label_key", args.label_key)
        writer.add_scalar("learning rate", args.lr, 0)
        writer.flush()
        logging.info(f"=> Model builder done, wrote model to '{model_path}'")
        plot_train_data(epoch_loss_values, metric_values, mean_val_acc, args.out_dir)

    predictions = []
    gts = []
    logging.info('Evaluating model')
    wrong_predictions = defaultdict(list)
    predictions_per_slide = defaultdict(lambda: np.array([], dtype=np.int32))
    gt_for_slide = {}
    with eval_mode(model):
        for item in tqdm(dl_test):
            y = model(item["image"].to(device))
            prob = F.softmax(y).detach().to("cpu")
            pred = torch.argmax(prob, dim=1).numpy()

            pred_positive = prob[:, 1].numpy() # only extract "positive", i.e. probability of being malignant
            predictions += list(labels[x] for x in pred)

            gt = item[CommonKeys.LABEL].detach().cpu().numpy()
            gts += list(labels[x] for x in gt)
            for i,(p,g) in enumerate(zip(pred, gt)):
                slide_name = os.path.basename(os.path.dirname(item["filename"][i]))
                gt_for_slide[slide_name] = g
                if args.label_key == "Sample Type":
                    predictions_per_slide[slide_name] = np.append(predictions_per_slide[slide_name], pred_positive[i])
                else:
                    predictions_per_slide[slide_name] = np.append(predictions_per_slide[slide_name], p)
                if p != g:
                    wrong_predictions[labels[g]].append((item["filename"][i], g, p))

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
                axes[col].set_title("pred: {}".format(labels[p]), fontsize=fontsize)
            else:
                axes[row][col].imshow(plt.imread(image_filename))
                axes[row][col].set_title("pred: {}".format(labels[p]))
            i += 1
    writer.add_figure("Wrong Predictions", fig, 0)

    plot_results(gts, predictions, labels, "Test data - Tile level", writer)

    # benign = negative
    # Calculate cross-entropy from predictions and gts
    if args.label_key == 'Sample Type':
        predictions_per_slide = {k: np.max(v) for k,v in predictions_per_slide.items()}
        predictions_per_slide = {k: int(v > 0.5) for k,v in predictions_per_slide.items()}
    else:
        # Find the value that occurs the most in each "v" in predictions_per_slide
        predictions_per_slide = {k: np.bincount(v).argmax() for k,v in predictions_per_slide.items()}

    a = np.array([[l,gt_for_slide[s]] for s,l in predictions_per_slide.items()])
    predictions = a[:,0].astype(np.int32)
    predictions = list(labels[x] for x in predictions)
    gts = a[:,1].astype(np.int32)
    gts = list(labels[x] for x in gts)

    plot_results(gts, predictions, labels, "Test data - Slide level", writer)
    #plt.show()

def plot_results(gts, predictions, labels, title, writer):
    if len(labels) == 2:
        roc = roc_curve(gts, predictions)
        # check if roc[0] or roc[1] contains nan
        if np.isnan(roc[0]).any() or np.isnan(roc[1]).any():
            logging.warning("ROC curve contains nan values - omitting from tensorboard")
        else:
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.plot(roc[0], roc[1])
            writer.add_figure(f"ROC Curve - {title}", plt.gcf(), 0)
            auc = roc_auc_score(gts, predictions)
            writer.add_scalar(f"AUC - {title}", auc, 0)
    else:
        logging.info("not plotting ROC curve as there are more than 2 classes")

    cm = confusion_matrix(gts, predictions, labels=labels)
    correct_classifications = sum([cm[i][i] for i in range(len(labels))])
    wrong_classifications = len(gts) - correct_classifications
    total = correct_classifications + wrong_classifications
    writer.add_scalar("Overall - " + title, correct_classifications / total, 0)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cmd = disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])
    fig = cmd.ax_.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(20)
    writer.add_figure(f"Confusion Matrix - {title}", cmd.figure_)

if __name__ == '__main__':
    main()
