#!/usr/bin/env python

# orig paper: adds two new layers for classification and re-trains
# https://github.com/TahDeh/TCGA_Acquisition_site_project/blob/main/tss-feature-extraction.ipynb
# guide
#https://snappishproductions.com/blog/2020/05/25/image-self-supervised-training-with-pytorch-lightning.html.html

import glob
import tempfile
import os
import sys

import numpy as np
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
from monai.handlers import StatsHandler, from_engine
from monai.engines import SupervisedTrainer
from sklearn.mixture import GaussianMixture
from monai.inferers import SimpleInferer
from monai.optimizers import generate_param_groups
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy
from monai.apps import MedNISTDataset

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--feature_extractor', default='./pretrained/checkpoint.pth.tar', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--tcga_annotation_file', default='./out/annotation/recurrence_annotation_tcga.pkl', type=str, help='path to TCGA annotations')
parser.add_argument('--src_dir', default='/Data/TCGA_LUSC/preprocessed/by_class/lung_scc', type=str, help='path to preprocessed slide images')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted embeddings')

args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    raise RuntimeError("cuda not found, use different comp :-(")
print(device)

def ensure_dir_exists(path):
    dest_dir = os.path.dirname(path)
    if not os.path.exists(dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)

def retrain_model(feature_extractor, dl, annotations):
    embedding_dict = defaultdict(list)
    feature_extractor.train()
    #subtype_model.eval()

    with torch.no_grad():
        for d in tqdm(dl, position=0, leave=True, desc="processing batch"):
            img_batch = d['image']
            bag_idx = d['slide_id']
            tile_idx = d['tile_id']

            feat = feature_extractor(img_batch.to(device))

            for f,bag,tile in zip(feat, bag_idx, tile_idx):
                embedding_dict[bag].append((f[np.newaxis, :].cpu().numpy(), tile))
                slide_id = "-".join(bag.split("-")[0:3])
                outcomes_dict[slide_id] = annotations[slide_id]
        # The next for loop is more about making tensors into numpy arrays. We prune away the first dimension which doesn't need to exist
        # it is not merging all tiles from slides
        for slide_id in embedding_dict:
            # flatten the array: np.concatenate(np.array([[1,2],[3,4]]), axis=0) = array([1, 2, 3, 4])
            embedding_dict[slide_id] = list(zip(np.concatenate([x[0] for x in embedding_dict[slide_id]], axis=0), [x[1] for x in embedding_dict[slide_id]]))
    # Embedding dict now has all tensors for each tiles with tumour, grouped by slide
    # Outcomes_dict has annotation info for all slides with tumorous tile, e.g.
    # ... {0: {'recurrence': 0, 'slide_id': ['TCGA-   ...
    return embedding_dict, outcomes_dict

def load_model(net, model_path):
    # original
    checkpoint = torch.load(model_path)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)
    #net.last_linear = nn.Identity() # the linear layer removes our dependency/link to the key encoder
    # i.e. we can write net(input) instead of net.encoder_q(input)

def attach_layers(model, num_neurons_in_layers, out_classes):
    model.last_linear = nn.Identity()
    return model

def main():
    annotations = {}
    tcga_annotation = pickle.load(open(args.tcga_annotation_file, 'rb')) if os.path.exists(args.tcga_annotation_file) else {}
    annotations = {**tcga_annotation}

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor)
    model = attach_layers(model, [500, 200], 35)
    model.to('cuda')
    feature_extractor = nn.DataParallel(model, device_ids=[0])

    all_data = []
    number_of_slides = len(glob.glob(f"{args.src_dir}{os.sep}*"))
    splits = [int(number_of_slides * 0.7), int(number_of_slides * 0.1), int(number_of_slides * 0.2)]
    def add_dir(directory):
        all_data = []
        for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
            if os.path.isfile(filename):
                slide_id = os.path.basename(filename.split(os.sep)[-2])
                tile_id = os.path.basename(filename.split(os.sep)[-1])
                all_data.append({"image": filename, "tile_id": filename, "slide_id": slide_id})
        return all_data
    train_data, val_data, test_data = [], [], []
    for i, directory in enumerate(glob.glob(f"{args.src_dir}{os.sep}*")):
        if i < splits[0]:
            train_data += add_dir(directory)
        elif i < splits[0] + splits[1]:
            val_data += add_dir(directory)
        else:
            test_data += add_dir(directory)

    transformations = mt.Compose(
        [
            mt.LoadImaged("image", image_only=True),
            mt.EnsureChannelFirstd("image"),
            mt.ToTensord("image", track_meta=False),
            # doesnt work?
            # mt.ToDeviceD(keys="image", device=device),
        ])

    # # note that we split the train data again, not the entire dataset
    model_name = condssl.builder.MoCo.__name__
    data_dir_name = list(filter(None, args.data_dir.split(os.sep)))[-1]
    out_path = os.path.join(args.out_dir, model_name, data_dir_name)
    model_filename = os.path.join(out_path, 'model', 'checkpoint_{}_{}_#NUM#_{}_m{}_n{}_o{}.pth.tar'.format(model_name, data_dir_name, args.condition, args.batch_size, args.batch_slide_num, args.batch_inst_num))

    train_data, val_data, test_data = build_file_list(args.data_dir, args.file_list_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = generate_param_groups(
        network=model,
        layer_matches=[lambda x: x.class_layers, lambda x: "conv.weight" in x[0]],
        match_types=["select", "filter"],
        lr_values=[1e-3, 1e-4],
    )

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    train_ds = MedNISTDataset(root_dir=root_dir, transform=transformations, section="training", download=True)

    # TODO: am I supposed to keep using the training data now, or just from test data?
    dl = DataLoader(dataset=Dataset(train_data, transformations), batch_size=64, num_workers=4, shuffle=True)
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=dl,
        network=model,
        optimizer=torch.optim.Adam(params, 1e-5),
        loss_function=torch.nn.CrossEntropyLoss(),
        inferer=SimpleInferer(),
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        train_handlers=StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    )
    retrain_model(model, dl, annotations)

    train_data, annotations = load_data("train_data_path")
    model = load("my_model")
    predictions = model.inference("test_data_path")

    analysis = monai.test_overfitting(model, train_data, annotations["confounder"], predictions)
    print(analysis)

if __name__ == "__main__":
    main()

def load_data(a):
    pass