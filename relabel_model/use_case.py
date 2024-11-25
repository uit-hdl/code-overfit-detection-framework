#!/usr/bin/env python
import argparse
import logging
import os
import sys

import pandas as pd
import torch
from torch import nn
from monai.networks.utils import freeze_layers

from misc.monai_boilerplate import init_tb_writer
from network.inception_v4 import InceptionV4
from relabel_model import finetune


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
    parser = argparse.ArgumentParser(description='Extract embeddings ')

    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--feature-extractor',
                        # default='./model_out2b1413ba2b3df0bcd9e2c56bdbea8d2c7f875d1e/MoCo/tiles/model/checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar',
                        default=os.path.join('model_dir', 'checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar'),
                        type=str, help='path to trained model')
    parser.add_argument('--label-file',
                        default=os.path.join('annotation', 'TCGA', 'luad-lusc-joined-sample.tsv'), type=str,
                        help='path to TCGA annotations')
    parser.add_argument('--label-key', default='Sample Type', type=str,
                        help='default key to use for doing fine-tuning. If set to "my_inst", will retrain using institution as label')
    parser.add_argument('--src-dir', default=os.path.join('Data', 'TCGA_LUSC', 'tiles'), type=str,
                        help='path to preprocessed slide images')
    parser.add_argument('--out-dir', default='./out', type=str, help='path to save extracted embeddings')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only',
                        dest='debug_mode')
    parser.add_argument('--lr', '--learning-rate', default=1e-05, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument(
        "--tensorboard-name",
        required=False,
        type=str,
        help="Name of the run, used for tracking stats. e.g. 'testing_no_cache', etc, or leave blank",
    )
    parser.add_argument(
        "--tensorboard-dir",
        required=False,
        type=str,
        default=None,
        help="Tensorboard outputdir. Defaults to /tmp/tb_$USER",
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    writer = init_tb_writer(os.path.join(args.out_dir, "tb_logs"), args.tensorboard_name,
                            {
                                "src-dir": args.src_dir,
                                "epochs": args.epochs,
                                "batch": args.batch_size,
                                "debug": str(args.debug_mode),
                            })


    labels = pd.read_csv(args.labels_file, sep=",", header=0)
    # set index to be the first column
    labels = labels.set_index(labels.columns[0])

    if args.debug_mode:
        limit = 64
        args.epochs = min(3, args.epochs)
        labels = labels[:limit]
        logging.warning(f"Debug mode enabled. Only using {limit} samples in train and validation sets")

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor, device)

    model = attach_layers(model, [500, 200], len(labels))
    freeze_layers(model, exclude_vars="last_linear")

    finetune.assess_model(model, labels[args.label_key], writer,
                          out_dir=args.out_dir,
                          lr=args.lr,
                          batch_size=args.batch_size)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
