#!/usr/bin/env python
import argparse
import logging
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from ignite.metrics import Accuracy, Loss
from monai.data import Dataset, DataLoader
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import CheckpointSaver, TensorBoardStatsHandler, from_engine, StatsHandler, ValidationHandler
from monai.inferers import SimpleInferer
from monai.transforms import EnsureTyped, Compose
from monai.utils import CommonKeys

from feature_extraction.overlap import ensure_dir_exists
from misc.monai_boilerplate import init_tb_writer, divide_data
from network.inception_v4 import InceptionV4
from time import time
from relabel_model.finetune import FinetuneManager, evaluate_model, construct_datasets



def load_zarr_store(store_path):
    import zarr

    with zarr.DirectoryStore(store_path) as store:
        root = zarr.open_group(store)
        logging.info(f"Loading embeddings from {store_path}")
        root_data = list(root.arrays(recurse=True))
        names, features = [x[1].name for x in root_data], [x[1][:] for x in root_data]
        logging.info(f"Finished loading embeddings")
        logging.info(root.info)
        return names, features


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train(out_dir, model, dl_train, dl_val, epochs, device, optimizer, loss, writer):
    ensure_dir_exists(out_dir)

    val_handlers = [
        CheckpointSaver(save_dir=out_dir, save_dict={"net": model}, epoch_level=True, save_interval=2),
        TensorBoardStatsHandler(writer, output_transform=lambda x: x),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=dl_val,
        network=model,
        val_handlers=val_handlers,
        key_val_metric={
            "val_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))
        },
        postprocessing=Compose([EnsureTyped(keys=CommonKeys.PRED)]),
    )

    train_handlers = [StatsHandler(tag_name="train_loss", output_transform=from_engine([CommonKeys.LOSS], first=True)),
                      ValidationHandler(1, evaluator),
                      TensorBoardStatsHandler(writer, output_transform=lambda x: x),
                      ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=epochs,
        train_data_loader=dl_train,
        network=model,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))},
        additional_metrics={"train_loss": Loss(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL], first=False), loss_fn=loss)},
        train_handlers=train_handlers,
    )

    trainer.run()

    return

def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings ')

    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--embeddings-path', default='out/inception_luad-tiles_embedding.zarr', type=str,
                        help="location of embedding zarr (e.g. from save_embeddings.py)")
    parser.add_argument('--distributed', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs')
    parser.add_argument('--label-file', type=str, help='path to file annotations')
    parser.add_argument('--balanced-roundup', type=int, default=100, help="Maximum number of samples to have from each class")
    parser.add_argument('--label-key', type=str, help='default key to use for doing fine-tuning. If not set, will use the first column in the label-file')
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


def main(evaluate_only=False):
    args = parse_args()
    run_name = args.tensorboard_name or str(time())
    writer = init_tb_writer(os.path.join(args.out_dir, "tb_logs"), run_name,
                            {
                                "src-dir": args.embeddings_path,
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
    class_map = {c: i for i, c in enumerate(labels.unique())}

    names, features = load_zarr_store(args.embeddings_path)
    all_data = [{CommonKeys.IMAGE: f, CommonKeys.LABEL: n, "filename": n} for n, f in zip(names, features)]
    splits = divide_data(all_data, balanced=True, balanced_roundup=args.balanced_roundup)
    dl_train = DataLoader(Dataset(splits['train']), drop_last=True, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(Dataset(splits['validation']), drop_last=True, batch_size=args.batch_size, shuffle=True)
    dl_test = DataLoader(Dataset(splits['test']), drop_last=True, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProbe(len(features[0]), len(class_map))
    model = model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DataParallel(model)

    model_dir = os.path.join(args.out_dir, "relabelled_models")
    if not evaluate_only:
        train(model_dir, model, dl_train, dl_val, args.epochs, device, torch.optim.Adam(model.parameters(), lr=args.lr), nn.CrossEntropyLoss(), writer)

    model_paths = [p for p in os.listdir(args.model_dir) if p.endswith(".pt")]
    for model_path in model_paths:
        model = LinearProbe(len(features[0]), len(class_map))
        model.load_state_dict(torch.load(os.path.join(args.model_dir, model_path), map_location="cpu"))

        evaluate_model(model, dl_test, class_map, writer)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    _evaluate_only = False
    main(_evaluate_only)

    logging.info("Done")
