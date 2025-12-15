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
import math
import os
import sys
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn.functional as F
import zarr
from IPython.utils.path import ensure_dir_exists
from lp_inspect import make_lp
from lp_inspect.model import LinearProbe
from monai.networks import eval_mode
from sklearn.metrics import accuracy_score, cohen_kappa_score
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

def dirichlet_sample(n):
    """Generates n random floats summing to 1 using Dirichlet distribution."""
    # A vector of ones defines the base distribution
    return np.random.dirichlet(np.ones(n))

def balanced_sample_dataset(df, subset_size) -> pd.DataFrame:
    """
    df: headers:
     filename,bcr_patient_barcode,institution,gender,race,tumor_stage,slide_id,disease
    """
    # get the institution that has the lowest number of samples for each stage.
    min_counts = {}
    stages = df["tumor_stage"].unique()
    for stage in stages:
        stage_counts = df[df["tumor_stage"] == stage].groupby("institution").size()
        min_counts[stage] = stage_counts.min()

     # toy sample: df has len 100 with 5 institutions.
    insts = df["institution"].unique()
    # that means we have 20 samples per institution
    samples_per_inst = math.floor(len(df) / len(insts))
    # if we're keeping 90%, we'll keep 18 of those 20 (dropping 2)
    dropped_per_inst = math.ceil(samples_per_inst - (samples_per_inst * subset_size))
    # of the 18, we want to balance per stage
    # lets say they were divided as 18 being Stage 1, 1 Stage II and 1 Stage III sample.
    # we need to draw out two. Lest say we do it from the last stage, e.g. (0, 0, 2)
    # this is an issue, since there were only 1 Stage III sample.
    # so we need to figure out the maximum number of samples...

    drop_list = {}
    sort_counts = sorted(min_counts.items(), key=lambda x: x[1])
    dropped_previous = 0

    # starting from the distribution with the lowest number of samples.
    for stage,max_to_drop in sort_counts[:-1]:
        how_many_to_drop = np.random.randint(low=0, high=min(max_to_drop , dropped_per_inst - dropped_previous) + 1)
        drop_list[stage] = how_many_to_drop
        dropped_previous += how_many_to_drop

    drop_list[sort_counts[-1][0]] = dropped_per_inst - sum(drop_list.values())

    new_dfs = []
    for inst in insts:
        for i, stage in enumerate(stages):
            stage_df = df[(df["tumor_stage"] == stage) & (df["institution"] == inst)]
            samples_to_drop = drop_list[stage]
            if samples_to_drop >= 0:
                stage_df = stage_df.sample(n=len(stage_df) - samples_to_drop, random_state=42)
            new_dfs.append(stage_df)
    new_df = pd.concat(new_dfs)

    return new_df


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune embeddings from a model using linear probe')

    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--subset-size', default=0.9, type=float, help="percentage of data to use for each CI computation")
    parser.add_argument('--embeddings-path', default="./", type=str, help="locations of embedding zarr (e.g. from save_embeddings.py)")
    parser.add_argument('--rounds', default=100, type=int, help='how many rounds to do for CI computation')
    parser.add_argument('--label-file', default="./balanced_dataset_top5.csv", type=str, help='path to file annotations')
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

    if args.label_file is None or not os.path.exists(args.label_file):
        raise ValueError(f"Label file '{args.label_file}' does not exist")
    labels = pd.read_csv(args.label_file, sep=",", header=0, dtype=defaultdict(lambda: str))
    # set index to be the first column
    labels = labels.set_index(labels.columns[0])

    label_key = args.label_key
    if not args.label_key:
        label_key = labels.columns[0]

    lookup_index = defaultdict(lambda: {})
    for fn, row in labels.iterrows():
        bn = os.path.basename(fn)
        dn = os.path.basename(os.path.dirname(fn))
        lookup_index[dn][bn] = row[label_key]

    embedding_set = []
    #test_division = balanced_sample_dataset(labels, 0.5)
    # count how many of each "institution" is in test_division
    #counts = test_division.groupby("institution").size()
    if args.debug_mode:
        mock_embedding = []
        args.rounds = min(5, args.rounds)
        labels = labels.sample(n=512)
        labels = labels.reset_index()
        for i in range(512):
            mock_inst = np.random.randint(0, 5)
            mock_stage = ["Stage I", "Stage II", "Stage III"][np.random.randint(0, 2)]
            filename = f"img_{i}.png"
            mock_embedding.append({
                "image": np.random.rand(64).astype(np.float32),
                "stage": mock_stage,
                "filename": filename,
                "label": mock_inst,
                "institution": str(mock_inst)})

            # set labels[i]['filename'] = filename
            labels.loc[i, 'filename'] = filename

        labels = labels.set_index(labels.columns[0])
        print("Debug mode! Using mock data")
        args.epochs = min(2, args.epochs)
        # get 256 labels
        logging.warning(f"Debug mode enabled. Only using 256 samples in train and validation sets")
        embedding_set = mock_embedding
    else:
        data = load_zarr_store(args.embeddings_path)
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
        logging.info(f"{args.embeddings_path}: only keeping data from annotation file: {len(new_data)} out of {len(data)} entries")
        if not new_data:
            raise ValueError(f"No data found in {args.label_file}")
        # sort new_data by "filename"
        new_data.sort(key=lambda x: x["filename"])
        embedding_set = new_data

    # 1-10
    # X
    # X + 2

    run_name = args.tensorboard_name or str(time())
    writer = init_tb_writer(os.path.join(args.out_dir, "lp_tb_logs"), run_name, extra=
    {
        "embeddings_path": args.embeddings_path,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "debug": str(args.debug_mode),
        "subset_size": args.subset_size,
        "rounds": args.rounds,
        "label-file": args.label_file,
        "label-key": label_key,
        "number_of_samples": len(embedding_set),
        "number_of_labels": len(labels),
    })
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ensure_dir_exists(args.out_dir)

    gt_li = []
    pred_li = []
    accuracies = []
    pred_ep = []
    gt_ep = []

    for n in range(args.rounds):
        sub_labels = balanced_sample_dataset(labels, subset_size=args.subset_size)
        data = []
        # only keep entries from data that is also in labels
        for ep in embedding_set:
            if ep["filename"] in sub_labels.index:
                new_ep = {'image': ep['image'],
                          'label': ep['label'],
                          'filename': ep['filename']
                          }
                data.append(new_ep)

        assert len(data) > 0, f"No data"
        writer.add_scalar("subset_size", len(data), global_step=n)


        dl_test, model, _ = make_lp(data=data,
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
                prob = F.softmax(y, dim=1).detach().to("cpu")
                pred = torch.argmax(prob, dim=1).numpy()
                preds = np.append(preds, pred)
                gt = item["label"].detach().cpu().numpy()
                gts = np.append(gts, gt)
        pred_ep.append(preds)
        gt_ep.append(gts)
        # compute accuracy between preds and gts
        accuracy = accuracy_score(gts, preds)
        accuracies.append(accuracy)

    kappa_scores = []
    for i,(p,g) in enumerate(zip(pred_ep, gt_ep)):
        kappa = cohen_kappa_score(p, g)
        kappa_scores.append(kappa)
        writer.add_scalar("kappa", kappa, global_step=i)

    for i,a in enumerate(accuracies):
        writer.add_scalar("accuracy", a, global_step=i)
        
    cl = 0.95  # confidence level
    ci = stats.t.interval(cl, df=len(accuracies) - 1, loc=np.mean(accuracies), scale=np.std(accuracies, ddof=1) / np.sqrt(len(accuracies)))
    print(accuracies)
    print(f"Accuracy ci={ci}, mean={np.mean(accuracies)}, std={np.std(accuracies, ddof=1)}")

    ci = stats.t.interval(cl, df=len(kappa_scores) - 1, loc=np.mean(kappa_scores), scale=np.std(kappa_scores, ddof=1) / np.sqrt(len(kappa_scores)))
    print(kappa_scores)
    print(f"kappa ci={ci}, mean={np.mean(kappa_scores)}, std={np.std(kappa_scores, ddof=1)}")


    logging.info("Inspect results with:\ntensorboard --logdir %s", os.path.join(args.out_dir, "lp_tb_logs"))
    logging.info("Done")
