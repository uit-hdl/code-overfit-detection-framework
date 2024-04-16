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

from monai.data import DataLoader, Dataset
from monai.utils import CommonKeys

from global_util import build_file_list, ensure_dir_exists

sys.path.append('./')

import condssl.builder
import argparse
import pickle
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from network.inception_v4 import InceptionV4
import monai.transforms as mt
from monai.inferers import SimpleInferer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

parser = argparse.ArgumentParser(description='Demographic parities')


#parser.add_argument('--feature_extractor', default='out/MoCo/tiles/model/relabelled_checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar/network_epoch=10.pt', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--feature_extractor', default='out/MoCo/tiles/model/relabelled_checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar/network_epoch=10.pt', type=str, help='path to feature extractor, which will extract features from tiles')
parser.add_argument('--tcga_annotation_file', default='./out/annotation/recurrence_annotation_tcga.pkl', type=str, help='path to TCGA annotations')
parser.add_argument('--clinical_path', default='./annotations/TCGA/clinical_tcga.tsv', type=str)
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

    clinicalTable = pd.read_csv(args.clinical_path, sep='\t').set_index('case_submitter_id')

    slide_annotations = pd.read_csv(args.slide_annotation_file, sep='\t', header=0)
    slide_annotations["my_slide_id"] = slide_annotations["File Name"].map(lambda s: s.split(".")[0])
    slide_annotations = slide_annotations.set_index("my_slide_id")
    tissue_types = slide_annotations["Sample Type"].unique().tolist()

    logging.info("Processing annotations")
    train_data, val_data, test_data = build_file_list(args.src_dir, args.file_list_path)
    # TODO: remove
    #print("warn: only using subset of data")
    #import random
    #random.shuffle(test_data)
    #test_data = test_data[:100]
    #for ds in [test_data]:
    for ds in [train_data, val_data, test_data]:
        for i in range(len(ds)):
            filename = ds[i]['filename']
            slide_id = os.path.basename(os.path.dirname(filename))
            patient_id = '-'.join(slide_id.split("-")[:3])
            clinical_row = clinicalTable.loc[patient_id]
            ann = slide_annotations.loc[slide_id]
            ds[i][CommonKeys.LABEL] = tissue_types.index(ann["Sample Type"])
            ds[i]["gender"] = clinical_row['gender']
    logging.info("Annotations complete")

    model = InceptionV4(num_classes=128)
    model = attach_layers(model, [500, 200], 2)
    logging.info(f"=> loading model '{args.feature_extractor}'")
    model.load_state_dict(torch.load(args.feature_extractor, map_location=device))
    logging.info('Model builder done')
    model.to(device)

    transformations = mt.Compose(
        [
            mt.LoadImaged("image", image_only=True),
            mt.EnsureChannelFirstd("image"),
            mt.ToTensord("image", track_meta=False),
            # doesnt work?
            mt.ToDeviceD(keys="image", device=device),
        ])

    inferer = SimpleInferer()
    dl_test = DataLoader(dataset=Dataset(test_data, transformations), batch_size=64, num_workers=0, shuffle=False, drop_last=False)
    y_pred = []
    y_true = []
    sf_data = []
    gender_data = []
    model.eval()
    logging.info("Beginning inference")
    with torch.no_grad():
        for item in tqdm(dl_test):
            pred = inferer(inputs=item[CommonKeys.IMAGE], network=model).squeeze()
            if pred.shape == (2,):
                am = pred.argmax().detach().cpu()
                y_pred.append(am.item())
                y_true.append(item[CommonKeys.LABEL].item())
                institutions = os.path.basename(os.path.dirname(item["filename"][0])).split("-")[1]
                sf_data.append(institutions)
                gender_data.append(item["gender"])
            else:
                am = pred.argmax(dim=1)
                y_pred += am.tolist()
                y_true += item[CommonKeys.LABEL].tolist()
                institutions = list(map(lambda f: os.path.basename(os.path.dirname(f)).split("-")[1], item["filename"]))
                sf_data += institutions
                gender_data += item["gender"]
    logging.info("Inference complete")

    for name,group in [("Institution", sf_data), ("Gender", gender_data)]:
        demographic_parity = demographic_parity_difference(y_true, y_pred, sensitive_features=group)
        equalized_odds = equalized_odds_difference(y_true, y_pred, sensitive_features=group)
        y_true_positive = []
        y_pred_for_positive = []
        sf_for_positive = []
        y_true_negative = []
        y_pred_for_negative = []
        sf_for_negative = []
        for yt,yp,sf in zip(y_true, y_pred, group):
            if yt == 1:
                y_true_positive.append(yt)
                y_pred_for_positive.append(yp)
                sf_for_positive.append(sf)
            else:
                y_true_negative.append(yt)
                y_pred_for_negative.append(yp)
                sf_for_negative.append(sf)

        if name == "Gender":
            ratio_males_in_positive = len(list(filter(lambda x: x == 'male', sf_for_positive))) / (len(sf_for_positive) / 100)
            print()
            print(f"In test data inference: {ratio_males_in_positive:.2f}% of primary tumor slides are from males ({100.0-ratio_males_in_positive:.2f}% females)")
            number_of_positives_for_males = len(list(filter(lambda x: x == 'male', sf_for_positive)))
            number_of_negatives_for_males = len(list(filter(lambda x: x == 'male', sf_for_negative)))
            total = number_of_positives_for_males + number_of_negatives_for_males
            ratio_positive_for_male = number_of_positives_for_males / (total / 100)
            print(f"Of {total} slides for males, {ratio_positive_for_male:.2f}% are primary tumors ({100.0-ratio_positive_for_male}% are benign)")
            ratio_females_in_positive = len(list(filter(lambda x: x == 'female', sf_for_positive))) / (len(sf_for_positive) / 100)
            number_of_positives_for_females = len(list(filter(lambda x: x == 'female', sf_for_positive)))
            number_of_negatives_for_females = len(list(filter(lambda x: x == 'female', sf_for_negative)))
            total = number_of_positives_for_females + number_of_negatives_for_females
            ratio_positive_for_female = number_of_positives_for_females / (total / 100)
            print(f"Of {total} slides for females, {ratio_positive_for_female:.2f}% are primary tumors ({100.0-ratio_positive_for_female:.2f}% are benign)")
            print()
        else:
            print()
            counts_positive = defaultdict(int)
            counts_negative = defaultdict(int)
            df = pd.DataFrame(data=set(group), columns=[name]).set_index(name)

            for inst,prediction in zip(group, y_pred):
                if prediction == 1:
                    counts_positive[inst] += 1
                else:
                    counts_negative[inst] += 1
            ratio_positives = {}
            overall_ratio_positive = {}
            overall_ratio_negative = {}
            for inst in set(group):
                total = counts_positive[inst] + counts_negative[inst]
                ratio_positives[inst] = counts_positive[inst] / (total / 100)
                overall_ratio_positive[inst] = float(counts_positive[inst]) / (len(sf_for_positive) / 100)
                overall_ratio_negative[inst] = float(counts_negative[inst]) / (len(sf_for_negative) / 100)

            df[tissue_types[1]] = ratio_positives
            df[tissue_types[0]] = {k: 100.0-v for k, v in ratio_positives.items()}
            df['Overall (pred) ratio positive'] = overall_ratio_positive
            df['Overall (pred) ratio negative'] = overall_ratio_negative

            counts_positive = defaultdict(int)
            counts_negative = defaultdict(int)
            for inst,prediction in zip(group, y_true):
                if prediction == 1:
                    counts_positive[inst] += 1
                else:
                    counts_negative[inst] += 1
            ratio_positives = {}
            overall_ratio_positive = {}
            overall_ratio_negative = {}
            overall_slide_count = {}
            for inst in set(group):
                total = counts_positive[inst] + counts_negative[inst]
                ratio_positives[inst] = counts_positive[inst] / (total / 100)
                overall_ratio_positive[inst] = float(counts_positive[inst]) / (len(sf_for_positive) / 100)
                overall_ratio_negative[inst] = float(counts_negative[inst]) / (len(sf_for_negative) / 100)
                overall_slide_count[inst] = (counts_positive[inst] + counts_negative[inst]) / ((len(sf_for_positive) + len(sf_for_negative)) / 100)

            df["GT " + tissue_types[1]] = ratio_positives
            df["GT " + tissue_types[0]] = {k: 100.0-v for k, v in ratio_positives.items()}
            df['Overall (GT) ratio positive'] = overall_ratio_positive
            df['Overall (GT) ratio negative'] = overall_ratio_negative

            df['Overall slide percentage'] = overall_slide_count

            print(df)
            out_path = os.path.join(args.out_dir, os.path.basename(os.path.dirname(args.feature_extractor)) + f"_{name}_distributions_fairness.csv")
            ensure_dir_exists(out_path)
            df.to_csv(out_path, index=True)
            # # collect normal vs primary tumor samples
            # # calculate the ratio of primary tumor samples
            # # calculate overall ratio
            pass



        equalized_opportunity = equalized_odds_difference(y_true_positive, y_pred_for_positive, sensitive_features=sf_for_positive)
        equalized_opportunity_negative = equalized_odds_difference(y_true_negative, y_pred_for_negative, sensitive_features=sf_for_negative)

        data = {"Demographic Parity": [demographic_parity],
                "Equalized Odds": [equalized_odds],
                f"Equalized Opportunity {tissue_types[1]}": [equalized_opportunity],
                f"Equalized Opportunity {tissue_types[0]}": [equalized_opportunity_negative],
                }
        df = pd.DataFrame(data)
        #out_path = os.path.join(args.out_dir, "fairness.csv")
        out_path = os.path.join(args.out_dir, os.path.basename(os.path.dirname(args.feature_extractor)) + f"_{name}_fairness.csv")
        ensure_dir_exists(out_path)
        df.to_csv(out_path, index=False)
        logging.info(f"Results (also written to {out_path}):")
        pd.set_option('display.width', None)
        print(df.to_string(index=False))

    overall_data = train_data + val_data + test_data
    males = list(filter(lambda x: x["gender"] == "male", overall_data))
    females = list(filter(lambda x: x["gender"] == "female", overall_data))
    male_distribution = len(list(filter(lambda x: x["gender"] == "male", overall_data))) / (len(overall_data) / 100)
    print(f"Of {len(overall_data)} patients, {len(males)} ({male_distribution:.2f}%) are male ({100.0-male_distribution}% female)")
    male_distribution_of_tumor = len(list(filter(lambda x: x[CommonKeys.LABEL] == tissue_types.index("Primary Tumor"), males))) / (len(males) / 100)
    female_distribution_of_tumor = len(list(filter(lambda x: x[CommonKeys.LABEL] == tissue_types.index("Primary Tumor"), females))) / (len(females) / 100)
    print(f"For males, {male_distribution_of_tumor}% of slides are primary tumors ({1.0-male_distribution_of_tumor}% are benign)")
    print(f"For females, {female_distribution_of_tumor}% of slides are primary tumors ({1.0-female_distribution_of_tumor}% are benign)")

    print("looking at the test data: ")
    males = list(filter(lambda x: x["gender"] == "male", test_data))
    females = list(filter(lambda x: x["gender"] == "female", test_data))
    male_distribution = len(list(filter(lambda x: x["gender"] == "male", test_data))) / (len(test_data) / 100)
    print(f"Of {len(test_data)} patients, {len(males)} ({male_distribution:.2f}%) are male ({100.0-male_distribution}% female)")
    male_distribution_of_tumor = len(list(filter(lambda x: x[CommonKeys.LABEL] == tissue_types.index("Primary Tumor"), males))) / (len(males) / 100)
    female_distribution_of_tumor = len(list(filter(lambda x: x[CommonKeys.LABEL] == tissue_types.index("Primary Tumor"), females))) / (len(females) / 100)
    print(f"For males, {male_distribution_of_tumor}% of slides are primary tumors ({1.0-male_distribution_of_tumor}% are benign)")
    print(f"For females, {female_distribution_of_tumor}% of slides are primary tumors ({1.0-female_distribution_of_tumor}% are benign)")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #get_logger("train_log")
    main()
