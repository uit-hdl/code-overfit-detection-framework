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
from scipy import stats as scistats
import torch.nn.functional as F
import zarr
from IPython.utils.path import ensure_dir_exists
from lp_inspect import make_lp
from lp_inspect.model import LinearProbe, evaluate_model
from monai.data import DataLoader, Dataset
from monai.networks import eval_mode
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm

from misc.monai_boilerplate import init_tb_writer


def get_dataset_from_labels(labels, label_key, embeddings_path, debug_mode=False):
    lookup_index = defaultdict(lambda: {})

    if not label_key:
        label_key = labels.columns[0]

    for fn, row in labels.iterrows():
        bn = os.path.basename(fn)
        dn = os.path.basename(os.path.dirname(fn))
        lookup_index[dn][bn] = row[label_key]

    if debug_mode:
        mock_embedding = []
        num_samples = 1000
        for i in range(num_samples):
            mock_inst = np.random.randint(0, 5)
            mock_stage = ["Stage I", "Stage II", "Stage III"][np.random.randint(0, 2)]
            filename = f"img_{i}.png"
            mock_embedding.append({
                "image": np.random.rand(64).astype(np.float32),
                "stage": mock_stage,
                # "filename": filename,
                "filename": labels.head(1).index[0],
                "label": mock_inst,
                "institution": str(mock_inst)})

        logging.warning(f"Debug mode enabled. Only using {num_samples} samples in train  sets")
        return mock_embedding

    data = load_zarr_store(embeddings_path)
    if not data:
        raise ValueError(f"No data found in {embeddings_path}")

    new_data = []
    for d in data:
        fn = d["filename"]
        bn = os.path.basename(fn)
        dn = os.path.basename(os.path.dirname(fn))
        if not (dn in lookup_index and bn in lookup_index[dn]):
            continue
        d["label"] = lookup_index[dn][bn]
        d["filename"] = os.path.join(os.getenv("PWD"), d["filename"].lstrip("/").replace("//", "/"))
        new_data.append(d)
    logging.info(
        f"{embeddings_path}: only keeping data from annotation file: {len(new_data)} out of {len(data)} entries")
    if not new_data:
        raise ValueError(f"No data found in labels")
    # sort new_data by "filename"
    new_data.sort(key=lambda x: x["filename"])
    return new_data


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

def random_sample_for_bootstrap(df) -> pd.DataFrame:
    # sample from df WITH repetition
    new_df = df.sample(frac=1, replace=True)
    return new_df


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune embeddings from a model using linear probe')

    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help=f'batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--embeddings-path', default="./", type=str, help="locations of embedding zarr (e.g. from save_embeddings.py)")
    parser.add_argument('--test-embeddings-path', default="./", type=str, help="locations of embedding zarr (e.g. from save_embeddings.py)")
    parser.add_argument('--rounds', default=100, type=int, help='how many rounds to do for CI computation')
    parser.add_argument('--label-file', default="out_lp/balanced_dataset_top5_train.csv", type=str, help='path to file annotations')
    parser.add_argument('--test-label-file', default="out_lp/balanced_dataset_top5_test.csv", type=str, help='path to file annotations')
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

def main():
    args = parse_args()

    if args.label_file is None or not os.path.exists(args.label_file):
        raise ValueError(f"Label file '{args.label_file}' does not exist")
    if args.test_label_file is None or not os.path.exists(args.test_label_file):
        raise ValueError(f"Testlabel file '{args.test_label_file}' does not exist")

    labels = pd.read_csv(args.label_file, sep=",", header=0, dtype=defaultdict(lambda: str))
    labels["filename"] = labels["filename"].str.replace("//", "/")
    labels = labels.set_index(labels.columns[0])

    test_labels = pd.read_csv(args.test_label_file, sep=",", header=0, dtype=defaultdict(lambda: str))
    test_labels["filename"] = test_labels["filename"].str.replace("//", "/")
    test_labels = test_labels.set_index(test_labels.columns[0])

    embedding_set = get_dataset_from_labels(labels, args.label_key, args.embeddings_path, args.debug_mode)
    embedding_set_test = get_dataset_from_labels(test_labels, args.label_key, args.test_embeddings_path, args.debug_mode)

    if args.debug_mode:
        args.epochs = min(2, args.epochs)
        args.rounds = min(5, args.rounds)
        embedding_set = embedding_set_test[:800]
        embedding_set_test = embedding_set_test[800:]
        # just use the first entry in test_labels for all the entries and repeat it 200 rows
        first_row = test_labels.head(1)
        test_labels = pd.concat([first_row] * 200)

    run_name = args.tensorboard_name or str(time())
    balanced = True
    writer = init_tb_writer(os.path.join(args.out_dir, "lp_tb_logs"), run_name, extra=
    {
        "embeddings_path": args.embeddings_path,
        "embeddings_path_test": args.test_embeddings_path,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "balanced": balanced,
        "debug": str(args.debug_mode),
        "rounds": args.rounds,
        "label-file": args.label_file,
        "test-label-file": args.test_label_file,
        "label-key": args.label_key,
        "number_of_samples": len(embedding_set),
        "number_of_labels": len(labels),
    })
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ensure_dir_exists(args.out_dir)

    val_ratio_for_lp = len(embedding_set_test) / len(embedding_set)
    train_ratio_for_lp = 1 - val_ratio_for_lp

    _, model, _, class_map = make_lp(data=embedding_set,
                                     out_dir=args.out_dir,
                                     balanced=balanced,
                                     writer=writer,
                                     epochs=args.epochs,
                                     eval_models=False,
                                     batch_size=args.batch_size,
                                     lr=args.lr,
                                     train_ratio=train_ratio_for_lp,
                                     val_ratio=val_ratio_for_lp,
                                     test_ratio=0.0)

    embedding_test_dir = os.path.dirname(args.test_embeddings_path)
    if not os.path.exists(os.path.join(args.out_dir, "subsets")):
        os.mkdir(os.path.join(args.out_dir, "subsets"))

    for i in range(args.rounds):
        dst_labels = os.path.join(args.out_dir, "subsets", f"balanced_dataset_top5_subset_{i}.csv")
        if os.path.exists(dst_labels):
            continue
        new_df = random_sample_for_bootstrap(test_labels)
        new_df.to_csv(dst_labels)

    accuracies = []
    kappas = []
    
    # Create a lookup for quick access to embeddings
    embedding_lookup = {entry["filename"]: entry for entry in embedding_set_test}

    for i in range(args.rounds):
        test_labels_for_run = pd.read_csv(os.path.join(args.out_dir, "subsets", f"balanced_dataset_top5_subset_{i}.csv"))
        test_labels_for_run["filename"] = test_labels_for_run["filename"].str.replace("//", "/")
        
        # Iterate over the labels index to preserve repetitions from the bootstrap
        embedding_set_test_for_run = []
        for fn in test_labels_for_run["filename"]:
            # Ensure the path matches the lookup key
            # We normalize the filename to match how it was stored in get_dataset_from_labels
            if fn in embedding_lookup:
                embedding_set_test_for_run.append(embedding_lookup[fn])
            else:
                # Alternative path normalization if needed
                norm_fn = "/" + fn.lstrip("/")
                if norm_fn in embedding_lookup:
                    embedding_set_test_for_run.append(embedding_lookup[norm_fn])
                else:
                    print(f"{fn} not found in embedding lookup")

        writer.add_scalar("test_size", len(embedding_set_test_for_run), i)
        dl_test = DataLoader(Dataset(embedding_set_test_for_run), batch_size=args.batch_size)
        class_map_inv = {v: k for k, v in class_map.items()}
        writer.add_scalar("num_labels", len(list(class_map.keys())), i)
        model_accuracy, kappa = evaluate_model(model, dl_test, os.path.join(args.out_dir, f"prediction_model_{i}.csv"), class_map=class_map_inv, device=device, writer=writer, step=args.epochs)
        accuracies.append(model_accuracy)
        kappas.append(kappa)

    accuracy_df = pd.DataFrame({
        "b": range(len(accuracies)),
        "accuracy": accuracies
    })
    accuracy_file=os.path.join(args.out_dir, "accuracies.csv")
    accuracy_df.to_csv(accuracy_file, index=False)
    print(f"Wrote accuracies to {accuracy_file}")

    cl = 0.95  # confidence level
    ma = np.mean(accuracies)
    ci = stats.t.interval(cl, df=len(accuracies) - 1, loc=ma,
                          scale=np.std(accuracies, ddof=1) / np.sqrt(len(accuracies)))
    print(accuracies)
    print(f"Accuracy ci={ci}, mean={np.mean(accuracies)}, std={np.std(accuracies, ddof=1)}")
    writer.add_scalar("accuracy_avg", ma)
    lb, ub = ci
    writer.add_scalar("accuracy_ci_lb", lb)
    writer.add_scalar("accuracy_ci_ub", ub)

    ci = stats.t.interval(cl, df=len(kappas) - 1, loc=np.mean(kappas),
                          scale=np.std(kappas, ddof=1) / np.sqrt(len(kappas)))
    print(kappas)
    mk = np.mean(kappas)
    print(f"kappa ci={ci}, mean={mk}, std={np.std(kappas, ddof=1)}")
    writer.add_scalar("kappa_avg", mk)
    lb, ub = ci
    writer.add_scalar("kappa_ci_lb", lb)
    writer.add_scalar("kappa_ci_ub", ub)

    logging.info("Inspect results with:\ntensorboard --logdir %s", os.path.join(args.out_dir, "lp_tb_logs"))


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    loggers = [logging.getLogger()]  # get the root logger
    loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    feature_inspect_logger = logging.getLogger("feature_inspect")
    feature_inspect_logger.setLevel(logging.WARNING)
    ignite_logger = logging.getLogger("ignite")
    ignite_logger.setLevel(logging.WARNING)

    main()

    logging.info("Done")
