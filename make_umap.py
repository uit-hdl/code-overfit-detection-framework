#!/usr/bin/env python
# coding: utf-8
'''
Make UMAP plots for the embeddings from a model
To make this example work, save a .zarr array with model embeddings stored in arrays to the path specified in the embeddings-path argument
Also provide a label-file with the labels for the embeddings, which has to have a format like:
filename,label_1,label_2,label_3
/absolute/path/to/img1.png,0,1,0
...

The output will be stored in the out-dir, which will contain the trained model and tensorboard logs

Example:
  python make_umap.py --embeddings-path out/phikon__embedding.zarr out/inception__embedding.zarr --label-file annotations/top5TSSfiles.csv --label-key slide_id --sizes 10000 100000
'''
import argparse
import logging
import os
import sys

import numpy.typing
from sklearn.metrics import accuracy_score, cohen_kappa_score
import torch.nn.functional as F
from collections import defaultdict
from time import time
from typing import List

import scipy.stats as stats

import numpy as np
import pandas as pd
import torch
import zarr
from IPython.utils.path import ensure_dir_exists

from umap_inspect.explore import make_umap
import logging

from monai.networks import eval_mode
from tqdm import tqdm

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
        all_data = [{"image": np.array(f[:]).astype(np.float32), "filename": n} for f, n in zip(features, names)]
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
    parser = argparse.ArgumentParser(description='Make UMAP plots from the embeddings from a model')

    parser.add_argument('--embeddings-path', default=[], nargs='+', type=str, help="locations of embedding zarr (e.g. from save_embeddings.py)")
    parser.add_argument('--sizes', default=[], nargs='+', type=int, help="how many points to use for each embedding. E.g. `10 1000` will produce two plots with 10 and 1000 points respectively.")
    parser.add_argument('--label-file', type=str, help='path to file annotations')
    parser.add_argument('--label-key', type=str, help='default key to use for doing fine-tuning. If not set, will use the first column in the label-file')
    parser.add_argument('--out-dir', default='./out', type=str, help='path to save model output and tensorboards')
    parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only',
                        dest='debug_mode')
    parser.add_argument(
        "--tensorboard-name",
        required=False,
        type=str,
        help="Name of the run, used for tracking stats. e.g. 'testing_no_cache', etc, or leave blank",
    )
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # feature_inspect_logger = logging.getLogger("feature_inspect")
    # feature_inspect_logger.setLevel(logging.WARNING)
    ignite_logger = logging.getLogger("ignite")
    ignite_logger.setLevel(logging.WARNING)
    args = parse_args()

    labels = pd.read_csv(args.label_file, sep=",", header=0, dtype=defaultdict(lambda: str))
    # set index to be the first column
    labels = labels.set_index(labels.columns[0])

    label_key = args.label_key
    if not args.label_key:
        label_key = labels.columns[0]

    if args.debug_mode:
        limit = 100
        args.epochs = min(2, args.epochs)
        # shuffle the order in labels
        labels = labels.sample(frac=1)
        labels = labels[:limit]
        logging.warning(f"Debug mode enabled. Only using {limit} samples in train and validation sets")
    labels = labels[label_key]
    lookup_index = defaultdict(lambda: {})
    for fn,val in labels.items():
        bn = os.path.basename(fn)
        dn = os.path.basename(os.path.dirname(fn))
        lookup_index[dn][bn] = val

    embedding_sets = []
    for ep in args.embeddings_path:
        data = load_zarr_store(ep)
        if not data:
            raise ValueError(f"No data found in {args.embeddings_path}")

        new_data = []
        for d in data:
            fn = d["filename"]
            bn = os.path.basename(fn)
            dn = os.path.basename(os.path.dirname(fn))
            if not (dn in lookup_index and bn in lookup_index[dn]):
                continue
            d["label"] = lookup_index[dn][bn]
            new_data.append(d)
        logging.info(f"{ep:} only keeping data from annotation file: {len(new_data)} out of {len(data)} entries")
        if not new_data:
            raise ValueError(f"No data found in {args.label_file}")
        # sort new_data by "filename"
        new_data.sort(key=lambda x: x["filename"])
        embedding_sets.append((ep, new_data))
    # mock_1 = [{"image": np.random.rand(args.sizes[0]).astype(np.float32), "filename": f"img_{i}.png", "label": np.random.randint(0, 3)} for i in range(256)]
    # embedding_sets.append(("mock1", mock_1))
    # for i,n in enumerate(args.sizes[1:]):
    #     mocky = [{"image": np.random.rand(n).astype(np.float32), "filename": mock_1[i]["filename"], "label": mock_1[i]["label"]} for i in range(256)]
    #     embedding_sets.append((f"mock{i}", mocky))

    run_name = args.tensorboard_name or str(time())
    writer = init_tb_writer(os.path.join(args.out_dir, "umap_tb_logs"), run_name, extra=
    {
        "embeddings_path": ','.join(args.embeddings_path),
        "debug": str(args.debug_mode),
    })
    class_map = {c: i for i, c in enumerate(labels)}
    class_map_inv = {i: c for i, c in enumerate(labels)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for n in args.sizes:
        max_index = len(embedding_sets[0][1])
        kept_indices : numpy.typing.NDArray[int] = np.random.choice(np.arange(0, max_index), size=min(n, max_index), replace=False)

        rounds = 1
        ensure_dir_exists(args.out_dir)
        gt_li = []
        pred_li = []
        for label,ep in embedding_sets:
            data = np.copy(ep)[kept_indices].tolist()

            assert len(data) > 0, f"No data"

            labels = pd.DataFrame([{"label": x["label"], "filename": x["filename"]} for x in data])


            # drop any entry in ep at given indices
            dst_dist = os.path.join(os.path.dirname(args.out_dir), os.path.basename(args.out_dir) + f"_{os.path.basename(label).split("_")[0]}")
            make_umap(values=[x["image"] for x in data],
                      labels=labels,
                    out_dir=dst_dist,
                    writer=writer,
                    do_ss=True,
                      render_html=True,
                      )

        logging.info("Inspect results with:\ntensorboard --logdir %s", os.path.join(args.out_dir, "tb_logs"))
        logging.info("Done")
