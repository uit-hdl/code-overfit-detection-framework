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

from global_util import build_file_list, ensure_dir_exists, institution_lookup

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

parser = argparse.ArgumentParser(description='Extract embeddings ')

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

def attach_layers(model, num_neurons_in_layers, out_classes):
    model.last_linear = nn.Sequential(
        nn.Linear(1536, 500),
        nn.ReLU(),
        nn.Linear(500, 200),
        nn.ReLU(),
        nn.Linear(200, out_classes),
    )
    return model

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
    for ds in [train_data, val_data, test_data]:
        for i in range(len(ds)):
            institution = institution_map(ds[i])
            ds[i]['label'] = institutions.index(institution)

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor, device)
    model = attach_layers(model, [500, 200], len(institutions))
    model.to(device)


    transformations = mt.Compose(
        [
            mt.LoadImaged("image", image_only=True),
            mt.EnsureChannelFirstd("image"),
            mt.ToTensord("image", track_meta=False),
            # doesnt work?
            # mt.ToDeviceD(keys="image", device=device),
        ])

    params = generate_param_groups(network=model, layer_matches=[lambda x: x.last_linear], match_types=["select"], lr_values=[1e-3])
    opt = torch.optim.Adam(params, 1e-5)

    def prepare_batch(batchdata, device, non_blocking):
        img, classes = batchdata["image"], batchdata["label"]
        return convert_tensor(img, device, non_blocking), convert_tensor(classes, device, non_blocking)

    dl_val = DataLoader(dataset=Dataset(val_data, transformations), num_workers=4, shuffle=True)
    val_postprocessing = Compose([Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)])
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=dl_val,
        network=model,
        postprocessing=val_postprocessing,
    )

    # TODO: am I supposed to keep using the training data now, or just from test data?
    batch_size = 64
    dl = DataLoader(dataset=Dataset(train_data, transformations), batch_size=batch_size, num_workers=4, shuffle=True)
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=dl,
        network=model,
        optimizer=opt,
        loss_function=torch.nn.CrossEntropyLoss(),
        inferer=SimpleInferer(),
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        train_handlers=[StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
                        ValidationHandler(1, evaluator, epoch_level = True),
                        CheckpointSaver(save_dir=args.out_dir, save_dict={'network': model}, save_interval=1)
                        ],
    )

    how_many_batches = len(train_data) // batch_size

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

    dl_test = DataLoader(dataset=Dataset(test_data, transformations), batch_size=1, num_workers=0, shuffle=False)
    max_items_to_print = 20 if len(dl_test.dataset) > 20 else len(dl_test.dataset)
    with eval_mode(model):
        for item in dl_test:
            prob = torch.squeeze(model(item["image"].to(device)).detach().to("cpu")).numpy()
            pred = institutions[prob.argmax()]
            gt = institutions[(item["label"].detach().item())]
            print(f"Class prediction is {pred}:{institution_lookup[pred]}. Ground-truth: {gt}:{institution_lookup[gt]}")
            max_items_to_print -= 1
            #if max_items_to_print == 0:
                #break

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
    plt.savefig(os.path.join(args.out_dir, 'relabelling_stats.png'))

    #analysis = monai.test_overfitting(model, train_data, annotations["confounder"], predictions)
    #print(analysis)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #get_logger("train_log")
    main()
