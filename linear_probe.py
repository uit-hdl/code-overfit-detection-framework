#!/usr/bin/env python
# coding: utf-8
'''
Fine-tune using the output from the embeddings of a model
To make this example work, save a .zarr array with model embeddings stored in arrays to the path specified in the embeddings-path argument
Also provide a label-file with the labels for the embeddings, which has to have a format like:
filename,label_1,label_2,label_3
/absolute/path/to/img1.png,0,1,0
...

The output will be stored in the out-dir, which will contain the trained model and tensorboard logs

Example:
  python examples/use_case_linear_probe.py --embeddings-path out/phikon_TCGA_LUSC-tiles_embedding.zarr \
    --label-file out/top5TSSfiles.csv \
    --out-dir out \
    --tensorboard-name tb_out \
    --epochs 1
'''
import argparse
import logging
import os
import sys
from collections import defaultdict
from time import time

import pandas as pd
import zarr
from IPython.utils.path import ensure_dir_exists

from lp_inspect import lp_eval
import logging

from misc.monai_boilerplate import init_tb_writer


def load_zarr_store(store_path):
    import zarr

    if not os.path.exists(store_path):
        raise ValueError(f"Store path {store_path} does not exist")

    with zarr.storage.LocalStore(store_path) as store:
        root = zarr.open_group(store)
        logging.info(f"Loading embeddings from {store_path}")
        names, features = find_all_arrays(root)
        if not names or not features:
            raise ValueError(f"No embeddings found in {store_path}")

        logging.info(f"Finished loading embeddings")
        logging.info(root.info)
        all_data = [{"image": f, "filename": n} for f, n in zip(features, names)]
        return all_data

def find_all_arrays(group):
    """
    Recursively iterate over a Zarr group to find all arrays.

    Parameters:
        group (zarr.Group): The current Zarr group.
        prefix (str): The prefix path for the current group.

    Yields:
        tuple: (full_path, array) for each Zarr array found.
    """
    groups_to_visit = [(group, "")]
    i = 0
    arrays = []
    names = []
    while True:
        if i >= len(groups_to_visit):
            break
        group, prefix = groups_to_visit[i]
        for name, member in group.groups():
            full_path = f"{prefix}/{name}"

            if isinstance(member, zarr.Group):
                groups_to_visit.append((member, full_path))
            for array_name, array in member.arrays():
                arrays.append(array)
                names.append(f"{full_path}/{array_name}")
        i += 1
    return (names, arrays)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune embeddings from a model using linear probe')

    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--embeddings-path', default=os.path.join('out', 'inception_TCGA_LUSC-tiles_embedding.zarr'), type=str,
                        help="location of embedding zarr (e.g. from save_embeddings.py)")
    parser.add_argument('--label-file', type=str, help='path to file annotations')
    parser.add_argument('--label-key', type=str, help='default key to use for doing fine-tuning. If not set, will use the first column in the label-file')
    parser.add_argument('--out-dir', default='./out', type=str, help='path to save model output and tensorboards')
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
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args()

    labels = pd.read_csv(args.label_file, sep=",", header=0, dtype=defaultdict(lambda: str))
    # set index to be the first column
    labels = labels.set_index(labels.columns[0])

    label_key = args.label_key
    if not args.label_key:
        label_key = labels.columns[0]

    if args.debug_mode:
        limit = 10 * args.batch_size
        args.epochs = min(2, args.epochs)
        # shuffle the order in labels
        labels = labels.sample(frac=1)
        labels = labels[:limit]
        logging.warning(f"Debug mode enabled. Only using {limit} samples in train and validation sets")
    labels = labels[label_key]
    lookup_index = labels.to_dict()
    data = load_zarr_store(args.embeddings_path)
    new_data = []
    for d in data:
        if d["filename"] not in lookup_index:
            continue
        d["label"] = lookup_index[d["filename"]]
        new_data.append(d)
    logging.info(f"only keeping data from annotation file: {len(new_data)} out of {len(data)} entries")
    data = new_data

    run_name = args.tensorboard_name or str(time())
    writer = init_tb_writer(os.path.join(args.out_dir, "lp_tb_logs"), run_name, extra=
    {
        "embeddings_path": args.embeddings_path,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "debug": str(args.debug_mode),
    })

    ensure_dir_exists(args.out_dir)
    lp_eval(data=data,
            out_dir=args.out_dir,
            balanced=True,
            writer=writer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr)

    logging.info("Inspect results with:\ntensorboard --logdir %s", os.path.join(args.out_dir, "tb_logs"))
    logging.info("Done")
