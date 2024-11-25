import glob
import logging
import os
import sys
import tempfile
from collections import defaultdict
from random import random
from time import time

import pandas as pd
from ignite.metrics import Accuracy, Loss
from matplotlib import pyplot as plt
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import StatsHandler, from_engine, ValidationHandler, CheckpointSaver, TensorBoardStatsHandler
from monai.handlers.tensorboard_handlers import SummaryWriter
from monai.inferers import SimpleInferer
from monai.transforms import Compose, EnsureTyped
from monai.utils import CommonKeys

from misc.global_util import ensure_dir_exists


def get_username():
    if os.environ.get("USER"):
        return os.environ.get("USER")

    login = None
    # this will typically fail in docker
    try:
        login = os.getlogin()
    except OSError:
        pass

    if not login or not login.isalnum():
        login = "unnamed"

    return login

def plot_distributions(data, mode, class_map, writer):
    count_per_label = pd.Series(data).value_counts()
    # convert the labels back to their original names (from integers)
    count_per_label.index = [class_map[x] for x in count_per_label.index]

    fig, ax = plt.subplots()
    ax.pie(count_per_label, labels=count_per_label.index, autopct='%1.1f%%')
    writer.add_figure(f"Label Distribution - {mode}", fig)

def divide_data(files, balanced=True, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    n = len(files)
    if balanced:
        # Group data by labels
        label_to_data = defaultdict(list)
        for item in files:
            label_to_data[item[CommonKeys.LABEL]].append(item)

        min_samples_per_label = min(len(items) for items in label_to_data.values())

        num_train = int(min_samples_per_label * train_ratio)
        num_val = int(min_samples_per_label * val_ratio)
        num_test = int(n * test_ratio)

        train_data, val_data, test_data = [], [], []

        for items in label_to_data.values():
            random.shuffle(items)
            train_data.extend(items[:num_train])
            val_data.extend(items[num_train:num_train + num_val])
            test_data.extend(items[num_train + num_val:num_train + num_val + num_test])
        if len(test_data) < n * test_ratio:
            ff = files.copy()
            random.shuffle(ff)
            test_data.extend(ff[len(test_data):int(n * test_ratio)])
        return {"train": train_data, "validation": val_data, "test": test_data}
    else:
        return {"train": files[:int(train_ratio * n)],
                "validation": files[int(train_ratio * n):int(1-val_ratio * n)],
                "test": files[int(1-test_ratio * n):]}



def build_file_list(data_dir, file_list_path, labels, label_key=None, balanced=False):
    if not os.path.exists(file_list_path):
        logging.info("File list not found. Creating file list in {}".format(file_list_path))
        all_data = []
        for filename in glob.glob(f"{data_dir}{os.sep}**", recursive=True):
            if os.path.isfile(filename) and filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                if labels is not None and not labels.empty and filename in labels.index:
                    all_data.append({'filename': filename, "label": labels.loc[filename][label_key]})
                else:
                    all_data.append({'filename': filename, "label": "unknown"})

        if labels is not None:
            splits = divide_data(all_data, balanced)
        else:
            splits = divide_data(all_data, False)

        if not splits["train"]:
            raise RuntimeError(f"Found no data in {data_dir}")

        train_data = pd.DataFrame(splits["train"])
        train_data["mode"] = "train"
        val_data = pd.DataFrame(splits["validation"])
        val_data["mode"] = "validation"
        test_data = pd.DataFrame(splits["test"])
        test_data["mode"] = "test"


        ensure_dir_exists(file_list_path)
        all_data = pd.concat([train_data, val_data, test_data])
        all_data.to_csv(file_list_path, index=False)

    all_data = pd.read_csv(file_list_path)
    train_data = all_data[all_data["mode"] == "train"].to_dict(orient='records')
    val_data = all_data[all_data["mode"] == "validation"].to_dict(orient='records')
    test_data = all_data[all_data["mode"] == "test"].to_dict(orient='records')
    for li in [train_data, val_data, test_data]:
        for item in li:
            item.pop("mode")
    logging.info(f"Loaded file list from {file_list_path}")

    train_data, val_data, test_data = [
        [
            {
                CommonKeys.IMAGE: entry['filename'],
                "filename": entry['filename'],
                CommonKeys.LABEL: entry[CommonKeys.LABEL]
            }
            for entry in data_list
        ]
        for data_list in [train_data, val_data, test_data]
    ]

    return train_data, val_data, test_data

def init_tb_writer(tb_dir, tb_name, extra):
    # 1. get name for tensorboard dst dir. trying to include username since that will ensure
    # that multiple people on one server won't cause write errors
    user = get_username()
    tb_dir = tb_dir or os.path.join(tempfile.gettempdir(), f"tb_{user}")
    tb_name = tb_name or str(time())
    tb_dst = os.path.join(tb_dir, tb_name)

    writer = SummaryWriter(log_dir=tb_dst)
    logging.info(
        f"Writing tensorboard stats to '{tb_dst}' (inspect with `tensoboard --logdir={tb_dst}`)"
    )
    try:
        writer.add_text("git_sha", os.popen("git rev-parse HEAD").read().strip())
    except Exception as e:
        logging.info(
            "could not get git SHA for tensorboard using `git rev-parse HEAD` - this message is safe to ignore"
        )

    writer.add_text("script_name", sys.argv[0])

    for key, val in extra.items():
        writer.add_text(key, str(val))

    return writer
