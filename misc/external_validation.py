#!/usr/bin/env python
# coding: utf-8
import glob
import logging
import os
import sys
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from monai.data import DataLoader, Dataset
from monai.networks import eval_mode

sys.path.append('./')
from misc.global_util import  ensure_dir_exists

import pandas as pd
import argparse
from monai.utils import  CommonKeys
import torch
import torch.nn as nn
from tqdm import tqdm
from network.inception_v4 import InceptionV4
import monai.transforms as mt
from monai.handlers.tensorboard_handlers import SummaryWriter
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--feature_extractor',
                    default='./out/MoCo/tiles/model/relabelled_checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar/network_epoch=10.pt',
                    #default='out/MoCo/tiles/model/relabelled_checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar/network_epoch=10.pt',
                    type=str, help='path to feature extractor, which will map tile to sample type')
parser.add_argument('--src-dir', default=os.path.join('Data', 'CPTAC', 'tiles'), type=str, help='path to preprocessed slide images')
parser.add_argument('--out-dir', default='./out', type=str, help='path to save extracted embeddings')
parser.add_argument('--label-key', default='sample_type', type=str, help='default key to use for getting labels')
parser.add_argument('--slide_annotation_file', default=os.path.join('annotations', 'CPTAC', 'slide.tsv'), type=str,
                    help='"Slide sheet", containing sample information, see README.md for instructions on how to get sheet')
parser.add_argument('--sample_annotation_file', default=os.path.join('annotations', 'CPTAC', 'sample.tsv'), type=str,
                    help='"Slide sheet", containing sample information, see README.md for instructions on how to get sheet')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only', dest='debug_mode')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')


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

def wrap_data(test_data, slide_annotations, labels, label_key, batch_size, workers):
    logging.info('Creating dataset')

    transformations = mt.Compose(
        [
            mt.LoadImaged([CommonKeys.IMAGE]),
            mt.EnsureChannelFirstd([CommonKeys.IMAGE]),
            mt.ToTensord([CommonKeys.IMAGE], track_meta=False),
            mt.EnsureTyped([CommonKeys.IMAGE, CommonKeys.LABEL], track_meta=False),
        ]
    )

    ds_test = Dataset(test_data, transformations)
    dl_test = DataLoader(ds_test, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=workers)

    logging.info("Dataset Created ...")
    logging.info("Number of images in dataset: {}".format(len(ds_test)))
    logging.info("Number of batches in DL: {}".format(len(dl_test)))
    return dl_test

def main():
    args = parser.parse_args()

    data_dir_name = list(filter(None, args.src_dir.split(os.sep)))[-1]
    out_path = os.path.join(args.out_dir, 'validation_' + data_dir_name, os.path.basename(os.path.dirname(args.feature_extractor)))

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
        logging.error("Annotation file not found: {}".format(args.tcga_annotation_file))
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info('Process annotations')
    slide_annotations = pd.read_csv(args.slide_annotation_file, sep='\t', header=0)
    slide_annotations = slide_annotations[['slide_submitter_id', 'sample_id']]

    sample_annotations = pd.read_csv(args.sample_annotation_file, sep='\t', header=0)
    sample_annotations = sample_annotations[['sample_id', 'sample_type']]

    df = slide_annotations.merge(sample_annotations, on='sample_id')
    df = df.set_index('slide_submitter_id')

    # TODO: I can consider to do percent_tumor_nuclei filtering

    labels = df[args.label_key].unique().tolist()
    labels.sort()
    labels.reverse()
    assert labels[0] == "Solid Tissue Normal" # making 0 the benign class if using 'sample type'

    logging.info('Creating dataset')
    all_data = []
    for filename in glob.glob(f"{args.src_dir}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(filename):
            sample_submitter_id = os.path.basename(os.path.dirname(filename))
            if sample_submitter_id in df.index:
                label = df.loc[sample_submitter_id]['sample_type']
                all_data.append({CommonKeys.IMAGE: filename, CommonKeys.LABEL: labels.index(label), "filename": filename})

    if args.debug_mode:
        logging.warning("Debug mode enabled!")
        all_data = all_data[:args.batch_size * 4]

    dl_test = wrap_data(all_data, slide_annotations, labels, args.label_key, args.batch_size, args.workers)

    model = InceptionV4(num_classes=128)
    model = attach_layers(model, [500, 200], len(labels))
    model.to(device)

    writer = SummaryWriter(log_dir=os.path.join(out_path, "runs"))
    if os.path.exists(args.feature_extractor):
        logging.info(f"=> loading model '{args.feature_extractor}'")
        model.load_state_dict(torch.load(args.feature_extractor, map_location=device))
        logging.info('Model builder done')

    predictions = []
    gts = []
    logging.info('Evaluating model')
    wrong_predictions = defaultdict(list)
    predictions_per_slide = defaultdict(lambda: np.array([], dtype=np.int32))
    gt_for_slide = {}
    with eval_mode(model):
        for item in tqdm(dl_test):
            y = model(item[CommonKeys.IMAGE].to(device))
            prob = F.softmax(y).detach().to("cpu")
            pred = torch.argmax(prob, dim=1).numpy()

            pred_positive = prob[:, 1].numpy() # only extract "positive", i.e. probability of being malignant
            predictions += list(pred)

            gt = item[CommonKeys.LABEL].detach().cpu().numpy()
            gts += list(gt)
            for i,(p,g) in enumerate(zip(pred, gt)):
                slide_name = os.path.basename(os.path.dirname(item["filename"][i]))
                gt_for_slide[slide_name] = g
                predictions_per_slide[slide_name] = np.append(predictions_per_slide[slide_name], pred_positive[i])
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
    if len(labels) == 2:
        predictions_per_slide = {k: np.max(v) for k,v in predictions_per_slide.items()}
        predictions_per_slide = {k: int(v > 0.5) for k,v in predictions_per_slide.items()}
    else:
        # Find the value that occurs the most in each "v" in predictions_per_slide
        predictions_per_slide = {k: np.bincount(v).argmax() for k,v in predictions_per_slide.items()}

    a = np.array([[l,gt_for_slide[s]] for s,l in predictions_per_slide.items()])
    predictions = list(a[:,0].astype(np.int32))
    gts = list(a[:,1].astype(np.int32))

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

    cm = confusion_matrix([labels[x] for x in gts], [labels[x] for x in predictions], labels=labels)
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
