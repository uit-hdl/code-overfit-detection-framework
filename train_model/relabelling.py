#!/usr/bin/env python

# orig paper: adds two new layers for classification and re-trains
# https://github.com/TahDeh/TCGA_Acquisition_site_project/blob/main/tss-feature-extraction.ipynb
# guide
#https://snappishproductions.com/blog/2020/05/25/image-self-supervised-training-with-pytorch-lightning.html.html
# other guide:
#https://github.com/Project-MONAI/tutorials/blob/main/modules/layer_wise_learning_rate.ipynb

import glob
import tempfile
import os
import sys

import numpy as np
from ignite.utils import convert_tensor
from monai.data import DataLoader, Dataset
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
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureTyped, AsDiscrete, EnsureType, Activations,
)

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--feature_extractor', default='./model_out2b1413ba2b3df0bcd9e2c56bdbea8d2c7f875d1e/MoCo/tiles/model/checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--tcga_annotation_file', default='./out/annotation/recurrence_annotation_tcga.pkl', type=str, help='path to TCGA annotations')
parser.add_argument('--file-list-path', default='./out/files.csv', type=str, help='path to list of file splits')
parser.add_argument('--src_dir', default='/Data/TCGA_LUSC/preprocessed/TCGA/tiles/', type=str, help='path to preprocessed slide images')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted embeddings')


def load_model(net, model_path):
    # original
    checkpoint = torch.load(model_path)
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
    for inst in list(map(institution_map, train_data)) + list(map(institution_map, val_data)):
        institutions.add(inst)
    institutions = sorted(institutions)
    for i in range(len(train_data)):
        institution = institution_map(train_data[i])
        train_data[i]['label'] = institutions.index(institution)
    for i in range(len(val_data)):
        institution = institution_map(val_data[i])
        val_data[i]['label'] = institutions.index(institution)

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor)
    model = attach_layers(model, [500, 200], len(institutions))
    model.to(device)


    # transform = Compose(
    #     [
    #         LoadImaged(keys="image"),
    #         EnsureChannelFirstd(keys="image"),
    #         ScaleIntensityd(keys="image"),
    #         EnsureTyped(keys="image"),
    #     ]
    # )
    # directory = os.environ.get("MONAI_DATA_DIRECTORY")
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    # train_ds = MedNISTDataset(root_dir=root_dir, transform=transform, section="training", download=True)

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

    act = Compose([EnsureType(), Activations(softmax=True)])
    toOnehot = Compose([EnsureType(), AsDiscrete(to_onehot=len(institutions))])
    def roc_auc_trans(batch):
        if isinstance(batch, list):
            aaa = [toOnehot(x["label"]).detach().cpu().unique() for x in batch]
            return [act(x["pred"]).detach().cpu() for x in batch], ([toOnehot(x["label"]).detach().cpu() for x in batch])
            #return torch.tensor([1 if x["pred"].argmax() == x["label"] else 0 for x in batch], device=device), torch.tensor([[1 if x["label"] == j else 0 for j in range(len(institutions))] for x in batch], device=device)

        return 1 if batch["pred"].argmax() == batch["label"]  else 0, [1 if batch["label"] == i else 0 for i in range(len(institutions))]

    def prepare_batch(batchdata, device, non_blocking):
        img, classes = batchdata["image"], batchdata["label"]
        return convert_tensor(img, device, non_blocking), convert_tensor(classes, device, non_blocking)

    batch_size = 64
    #dl_val = DataLoader(dataset=Dataset([val_data[0], val_data[-1]], transformations), batch_size=batch_size, num_workers=4, shuffle=True)
    dl_val = DataLoader(dataset=Dataset(val_data, transformations), batch_size=batch_size, num_workers=4, shuffle=True)
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=dl_val,
        network=model,
        #postprocessing=roc_auc_trans,
        key_val_metric={"rocauc": ROCAUC(output_transform=roc_auc_trans)},
        prepare_batch=prepare_batch,
    )

    # TODO: am I supposed to keep using the training data now, or just from test data?
    dl = DataLoader(dataset=Dataset(train_data[:64], transformations), batch_size=batch_size, num_workers=4, shuffle=True)
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=2,
        train_data_loader=dl,
        network=model,
        optimizer=opt,
        loss_function=torch.nn.CrossEntropyLoss(),
        inferer=SimpleInferer(),
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        train_handlers=[StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
                        ValidationHandler(1, evaluator),
                        CheckpointSaver(save_dir=args.out_dir, save_dict={'network': model}, save_interval=1)
                        ],
    )

    how_many_batches = len(train_data) // batch_size

    iterLosses = []
    batchSizes = []
    epochLossValues = []
    metricValues = []

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training(engine):
        batch_loss = engine.state.output
        loss = np.average([o["loss"] for o in engine.state.output])
        batch_len = len(engine.state.batch[0])
        lr = opt.param_groups[0]['lr']
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration

        iterLosses.append(loss)
        batchSizes.append(batch_len)

        print(f"Epoch {e}/{n} : {i}/{how_many_batches}, lr: {lr}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        # the overall average loss must be weighted by batch size
        overallAverageLoss = np.average(iterLosses, weights=batchSizes)
        epochLossValues.append(overallAverageLoss)

        # clear the contents of iter_losses and batch_sizes for the next epoch
        del iterLosses[:]
        del batchSizes[:]

        # reset iteration for next epoch
        engine.state.iteration = 0

        # fetch and report the validation metrics
        roc = evaluator.state.metrics["rocauc"]
        metricValues.append(roc)
        print(f"evaluation for epoch {engine.state.epoch},  rocauc = {roc:.4f}")

    trainer.run()

    pass
    a = 1
    #analysis = monai.test_overfitting(model, train_data, annotations["confounder"], predictions)
    #print(analysis)

if __name__ == "__main__":
    main()
