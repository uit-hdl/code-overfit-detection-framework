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

from monai.data import DataLoader, Dataset

from global_util import build_file_list, ensure_dir_exists

sys.path.append('./')

import condssl.builder
import argparse
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from network.inception_v4 import InceptionV4
import monai.transforms as mt
from monai.inferers import SimpleInferer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

parser = argparse.ArgumentParser(description='Demographic parities')

parser.add_argument('--feature_extractor', default='./model_out2b1413ba2b3df0bcd9e2c56bdbea8d2c7f875d1e/MoCo/tiles/model/checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--tcga_annotation_file', default='./out/annotation/recurrence_annotation_tcga.pkl', type=str, help='path to TCGA annotations')
parser.add_argument('--cptac_annotation_file', default='./out/annotation/recurrence_annotation_cptac.pkl', type=str, help='path to CPTAC annotations')
parser.add_argument('--slide_annotation_file', default='annotations/slide_label/gdc_sample_sheet.2023-08-14.tsv', type=str, help='"Sample sheet" from TCGA, see README.md for instructions on how to get sheet')
parser.add_argument('--file-list-path', default='./out/files.csv', type=str, help='path to list of file splits')
parser.add_argument('--src_dir', default='/Data/TCGA_LUSC/preprocessed/TCGA/tiles/', type=str, help='path to preprocessed slide images')
parser.add_argument('--out_dir', default='./out', type=str, help='path to save extracted embeddings')


def load_model(net, model_path, device):
    # original
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = {k.replace("encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)
    net.to(device)
    #net.last_linear = nn.Identity() # the linear layer removes our dependency/link to the key encoder
    # i.e. we can write net(input) instead of net.encoder_q(input)


def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # note that we split the train data again, not the entire dataset
    data_dir_name = list(filter(None, args.src_dir.split(os.sep)))[-1]
    model_filename = os.path.join(args.out_dir, condssl.builder.MoCo.__name__, data_dir_name, 'model', 'relabelled_{}'.format(os.path.basename(args.feature_extractor)))

    slide_annotations = pd.read_csv(args.slide_annotation_file, sep='\t', header=0)
    slide_annotations["my_slide_id"] = slide_annotations["File Name"].map(lambda s: s.split(".")[0])
    slide_annotations = slide_annotations.set_index("my_slide_id")
    tissue_types = slide_annotations["Sample Type"].unique().tolist()

    annotations = {}
    tcga_annotation = pickle.load(open(args.tcga_annotation_file, 'rb')) if os.path.exists(args.tcga_annotation_file) else {}
    cptac_annotation = pickle.load(open(args.cptac_annotation_file, 'rb')) if os.path.exists(args.cptac_annotation_file) else {}
    # TODO: CPTAC
    # annotations = {**tcga_annotation, **cptac_annotation}
    annotations = {**tcga_annotation}

    logging.info("Processing annotations")
    # TODO: this is slow. Can speed up the lookups?
    train_data, val_data, test_data = build_file_list(args.src_dir, args.file_list_path)
    for ds in [train_data, val_data, test_data]:
        for i in range(len(ds)):
            filename = ds[i]['filename']
            slide_id = os.path.basename(os.path.dirname(filename))
            ann = slide_annotations.loc[slide_id]
            ds[i]['label'] = tissue_types.index(ann["Sample Type"])
    logging.info("Annotations complete")

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor, device)

    transformations = mt.Compose(
        [
            mt.LoadImaged("image", image_only=True),
            mt.EnsureChannelFirstd("image"),
            mt.ToTensord("image", track_meta=False),
            # doesnt work?
            mt.ToDeviceD(keys="image", device=device),
        ])

    inferer = SimpleInferer()
    # TODO: remove
    print("warn: only using subset of data")
    dl_test = DataLoader(dataset=Dataset(test_data[:65], transformations), batch_size=64, num_workers=0, shuffle=False, drop_last=False)
    y_pred = []
    y_true = []
    sf_data = []
    model.eval()
    logging.info("Beginning inference")
    with torch.no_grad():
        for item in tqdm(dl_test):
            pred = inferer(inputs=item["image"], network=model).squeeze()
            if pred.shape == (128,):
                am = pred.argmax().detach().cpu()
                y_pred.append(am.item() % 2)
                y_true.append(item["label"].item())
                institutions = os.path.basename(os.path.dirname(item["filename"][0])).split("-")[1]
                sf_data.append(institutions)
            else:
                am = pred.argmax(dim=1)
                # TODO: update with model that has two output classes only
                y_pred += list(map(lambda l: l % 2, am.tolist()))
                y_true += item["label"].tolist()
                institutions = list(map(lambda f: os.path.basename(os.path.dirname(f)).split("-")[1], item["filename"]))
                sf_data += institutions
    logging.info("Inference complete")

    demographic_parity = demographic_parity_difference(y_true, y_pred, sensitive_features=sf_data)
    equalized_odds = equalized_odds_difference(y_true, y_pred, sensitive_features=sf_data)
    y_true_positive = []
    y_pred_for_positive = []
    sf_for_positive = []
    y_true_negative = []
    y_pred_for_negative = []
    sf_for_negative = []
    for yt,yp,sf in zip(y_true, y_pred, sf_data):
        if yt == 1:
            y_true_positive.append(yt)
            y_pred_for_positive.append(yp)
            sf_for_positive.append(sf)
        else:
            y_true_negative.append(yt)
            y_pred_for_negative.append(yp)
            sf_for_negative.append(sf)


    equalized_opportunity = equalized_odds_difference(y_true_positive, y_pred_for_positive, sensitive_features=sf_for_positive)
    equalized_opportunity_negative = equalized_odds_difference(y_true_negative, y_pred_for_negative, sensitive_features=sf_for_negative)
    # write demographic parity, equalized_odds and equalized_opportunity to csv

    data = {"Demographic Parity": [demographic_parity],
            "Equalized Odds": [equalized_odds],
            f"Equalized Opportunity {tissue_types[1]}": [equalized_opportunity],
            f"Equalized Opportunity {tissue_types[0]}": [equalized_opportunity_negative],
            }
    df = pd.DataFrame(data)
    out_path = os.path.join(args.out_dir, "fairness.csv")
    ensure_dir_exists(out_path)
    df.to_csv(out_path)
    logging.info(f"Results (also written to {out_path}):")
    pd.set_option('display.width', None)
    print(df)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #get_logger("train_log")
    main()
