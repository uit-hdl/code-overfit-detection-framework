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

from lp_inspect import make_lp
from lp_inspect.model import LinearProbe
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
    parser = argparse.ArgumentParser(description='Fine-tune embeddings from a model using linear probe')

    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--embeddings-path', default=[], nargs='+', type=str, help="locations of embedding zarr (e.g. from save_embeddings.py)")
    parser.add_argument('--embeddings-path-1', default=os.path.join('out', 'inception_TCGA_LUSC-tiles_embedding.zarr'), type=str,
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
    loggers = [logging.getLogger()]  # get the root logger
    loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    feature_inspect_logger = logging.getLogger("feature_inspect")
    feature_inspect_logger.setLevel(logging.WARNING)
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
        limit = 10 * args.batch_size
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

    # create two mock datasets that could have been produced by the code above, each with 256 entries
    # mock_1 = [{"image": np.random.rand(256).astype(np.float32), "filename": f"img_{i}.png", "label": np.random.randint(0, 3)} for i in range(256)]
    # mock_2 = [{"image": np.random.rand(256).astype(np.float32), "filename": mock_1[i]["filename"], "label": mock_1[i]["label"]} for i in range(256)]
    # embedding_sets.append(("mock 1", mock_1))
    # embedding_sets.append(("mock 2", mock_2))

    # verify that each set in embedding_sets have the exact same "filename" entries
    # ... only have to this once for sanity checks
    # for i, ep in enumerate(embedding_sets):
    #     ep_filenames = [d["filename"] for d in ep]
    #     for j, ep2 in enumerate(embedding_sets):
    #         if i == j:
    #             continue
    #         ep2_filenames = [d["filename"] for d in ep2]
    #         if ep_filenames != ep2_filenames:
    #             raise ValueError(f"embedding_sets[{i}] and embedding_sets[{j}] have different filenames")

    run_name = args.tensorboard_name or str(time())
    writer = init_tb_writer(os.path.join(args.out_dir, "lp_tb_logs"), run_name, extra=
    {
        "embeddings_path": ','.join(args.embeddings_path),
        "epochs": args.epochs,
        "batch": args.batch_size,
        "debug": str(args.debug_mode),
    })
    class_map = {c: i for i, c in enumerate(labels)}
    class_map_inv = {i: c for i, c in enumerate(labels)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rounds = 3
    lb = 0; ub = len(embedding_sets[0])
    # randomly sample 10% of numbers from lb to ub
    dropped_indices : List[np.ndarray] = []
    for n in range(rounds):
        indices = np.random.choice(np.arange(lb, ub), size=int(0.1 * len(embedding_sets[0])), replace=False)
        dropped_indices.append(indices)

    ensure_dir_exists(args.out_dir)
    gt_li = []
    pred_li = []
    for label,ep in embedding_sets:
        accuracies = []
        pred_ep = []
        gt_ep = []
        for n in range(rounds):
            # drop any entry in ep at given indices
            data = np.delete(np.copy(ep), dropped_indices[n])

            assert len(data) > 0, f"No data"

            dl_test, model, _ = make_lp(data=data.tolist(),
                    out_dir=args.out_dir,
                    balanced=True,
                    writer=writer,
                    epochs=args.epochs,
                    eval_models = False,
                    batch_size=args.batch_size,
                    lr=args.lr)
            preds = np.array([])
            gts = np.array([])
            with eval_mode(model):
                for item in tqdm(dl_test):
                    y = model(item["image"].to(device))
                    prob = F.softmax(y).detach().to("cpu")
                    pred = torch.argmax(prob, dim=1).numpy()
                    preds = np.append(preds, pred)
                    gt = item["label"].detach().cpu().numpy()
                    gts = np.append(gts, gt)
            pred_ep.append(preds)
            gt_ep.append(gts)
            # compute accuracy between preds and gts
            accuracy = accuracy_score(gts, preds)
            accuracies.append(accuracy)

        # compute 95% conf
        cl = 0.95  # confidence level
        print(accuracies)
        print(accuracies)
        ci = stats.t.interval(cl, df=len(accuracies) - 1, loc=np.mean(accuracies), scale=np.std(accuracies, ddof=1) / np.sqrt(len(accuracies)))
        print(f"{label}: ci={ci}, mean={np.mean(accuracies)}, std={np.std(accuracies, ddof=1)}")

        pred_li.append(pred_ep)
        gt_li.append(gt_ep)

    if len(pred_li) == 2:
        #compute kohens kappa between the two pred_li entries
        for d1, d2 in zip(pred_li[0], pred_li[1]):
            kappa = cohen_kappa_score(d1, d2)
            print(f"kappa between predictors={kappa}")
            writer.add_scalar("kappa", kappa, global_step=0)
    for l,p,g in zip([l[0] for l in embedding_sets], pred_li, gt_li):
        for d1, d2 in zip(p, g):
            kappa = cohen_kappa_score(d1, d2)
            print(f"kappa between {l} and gt={kappa}")
            writer.add_scalar("kappa", kappa, global_step=0)



    logging.info("Inspect results with:\ntensorboard --logdir %s", os.path.join(args.out_dir, "tb_logs"))
    logging.info("Done")
