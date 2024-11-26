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
from time import time
from relabel_model.finetune import FinetuneManager, evaluate_model, construct_datasets


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

def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings ')

    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--feature-extractor',
                        # default='./model_out2b1413ba2b3df0bcd9e2c56bdbea8d2c7f875d1e/MoCo/tiles/model/checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar',
                        default=os.path.join('model_dir', 'checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar'),
                        type=str, help='path to trained model')
    parser.add_argument('--distributed', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs')
    parser.add_argument('--label-file', type=str, help='path to file annotations')
    parser.add_argument('--model-dir', type=str, default=os.path.join("out", "relabelled"), help='path to models, if evaluating multiple models')
    parser.add_argument('--balanced-roundup', type=int, default=100, help="Maximum number of samples to have from each class")
    parser.add_argument('--label-key', type=str, help='default key to use for doing fine-tuning. If not set, will use the first column in the label-file')
    parser.add_argument('--src-dir', default=os.path.join('Data', 'TCGA_LUSC', 'tiles'), type=str,
                        help='path to preprocessed slide images')
    parser.add_argument('--out-dir', default='./out', type=str, help='path to save extracted embeddings')
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
    return parser.parse_args()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_name = args.tensorboard_name or str(time())
    writer = init_tb_writer(os.path.join(args.out_dir, "tb_logs"), run_name,
                            {
                                "src-dir": args.src_dir,
                                "epochs": args.epochs,
                                "batch": args.batch_size,
                                "debug": str(args.debug_mode),
                            })

    labels = pd.read_csv(args.label_file, sep=",", header=0)
    # set index to be the first column
    labels = labels.set_index(labels.columns[0])

    label_key = args.label_key
    if not args.label_key:
        label_key = labels.columns[0]

    if args.debug_mode:
        limit = 10 * args.batch_size
        args.epochs = min(2, args.epochs)
        labels = labels[:limit]
        logging.warning(f"Debug mode enabled. Only using {limit} samples in train and validation sets")
    labels = labels[label_key]

    model = InceptionV4(num_classes=128)
    load_model(model, args.feature_extractor, device)

    model = attach_layers(model, [500, 200], len(labels.unique()))
    freeze_layers(model, exclude_vars="last_linear")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DataParallel(model)

    finetune = FinetuneManager(model,
                               labels,
                               writer,
                               device,
                               args.epochs,
                               balanced=True,
                               balanced_roundup=args.balanced_roundup,
                               out_dir=os.path.join(args.out_dir, "relabelled", run_name),
                               lr=args.lr,
                               batch_size=args.batch_size
                               )
    finetune.finetune_model()

    # TODO: create a separate evaluation for dinobloom

def evaluate_inception(args):
    labels = pd.read_csv(args.label_file, sep=",", header=0)
    labels = labels.set_index(labels.columns[0])

    label_key = args.label_key
    if not args.label_key:
        label_key = labels.columns[0]
    labels = labels[label_key]

    # this will not have the same split as during training
    class_map, _, _, dl_test = construct_datasets(labels, args.batch_size)

    model_paths = [p for p in os.listdir(args.model_dir) if p.endswith(".pt")]
    if not model_paths:
        logging.error(f"No model files found in {args.model_dir}")
        return

    for model_path in model_paths:
        model = InceptionV4(num_classes=128)
        model = attach_layers(model, [500, 200], len(labels.unique()))
        model.load_state_dict(torch.load(os.path.join(args.model_dir, model_path), map_location="cpu"))
        load_model(model, args.feature_extractor, "cpu")

        writer = init_tb_writer(os.path.join(args.out_dir, "tb_logs"), model_path,
                                {
                                    "src-dir": args.src_dir,
                                })
        evaluate_model(model, dl_test, class_map, writer)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args()
    main(args)
    #evaluate_inception(args)
    logging.info("Done")
