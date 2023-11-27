import argparse
import glob
import os.path
import random
import csv
import tempfile
import sys
import time
import warnings
from collections import defaultdict

import monai.utils
from sklearn.model_selection import train_test_split

import monai.transforms as mt
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from monai.data import DataLoader, Dataset, set_track_meta, CacheDataset, PersistentDataset, SmartCacheDataset, ImageDataset, CacheNTransDataset
from torch.utils.data import Sampler, DistributedSampler

import samplers
sys.path.append("../")
import condssl.builder
import condssl.loader
import numpy as np
from global_util import build_file_list, ensure_dir_exists

from network.inception_v4 import InceptionV4
from train_util import *
import nvtx
import monai.utils
from monai.utils import Range
import contextlib
no_profiling = contextlib.nullcontext()

from itertools import zip_longest



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--profile', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='P', help='whether to profile training or not', dest='is_profiling')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--local_rank", type=int, default=0)


# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true', default=True, help='use cosine lr schedule')
parser.add_argument('--data-dir', default='./data/', type=str,
                    help='path to source directory')
parser.add_argument('--out-dir', default='./out/models/', type=str,
                    help='path to output directory')
parser.add_argument('--file-list-path', default='./out/files.csv', type=str, help='path to list of file splits')
parser.add_argument('--batch_slide_num', default=4, type=int)
parser.add_argument('--batch_inst_num', default=0, type=int)
parser.add_argument('--condition', default=True, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='C', help='whether to use conditional sampling or not', dest='condition')

def save_data_to_csv(data, filename, label):
    ensure_dir_exists(filename)
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["epoch", label])
        for e, l in enumerate(data):
            csvwriter.writerow([e,l])


def reset_seeds():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def train(train_loader, val_loader, model, criterion, optimizer, max_epochs, lr, cos, schedule, out_path, model_filename, is_profiling, is_distributed):
    epoch_loss_values = []
    accuracy1_values = []
    accuracy5_values = []
    metric_values = []
    epoch_times = []
    total_start = time.time()
    set_track_meta(True)
    model_savename = ""
    step = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch}/{max_epochs}")
        model.train()
        epoch_loss = 0
        acc5 = 0
        acc1 = 0
        train_loader_iterator = iter(train_loader)
        adjust_learning_rate(optimizer, epoch, lr, cos, schedule, max_epochs)

        # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_training_tutorial.ipynb
        # using step instead of iterate through train_loader directly to track data loading time
        # steps are 1-indexed for printing and calculation purposes
        #for i, d in enumerate(train_loader):
        for step in range(1, 3):
            if is_distributed:
                train_loader.batch_sampler.set_epoch(epoch)
            with Range("Step"):
                step_start = time.time()
                # profiling: train dataload
                # Download https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-3 to visualize
                with Range("Dataload") if is_profiling else no_profiling:
                    batch_data = next(train_loader_iterator)
                    images_q, images_k = batch_data['q'], batch_data['k']
                with Range("forward") if is_profiling else no_profiling:
                    output, target = model(im_q=images_q, im_k=images_k)
                    target = target.cpu()
                    loss = criterion(output, target)

                with Range("accuracy") if is_profiling else no_profiling:
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    acc1, acc5 = acc1[0], acc5[0]

                # compute gradient and do SGD step
                optimizer.zero_grad()
                with Range("backward_loss") if is_profiling else no_profiling:
                    loss.backward()
                with Range("update_optimizer") if is_profiling else no_profiling:
                    optimizer.step()

                epoch_loss += loss.item()
                acc5 += acc5
                acc1 += acc1
                print(
                        f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f} acc1: {acc1:.2f} acc5: {acc5:.2f} step time: {(time.time() - step_start):.4f}"
                )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        acc5 /= step
        accuracy5_values.append(acc5)
        acc1 /= step
        accuracy1_values.append(acc1)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")

        if not epoch % 20 or epoch == max_epochs:
            model_savename = model_filename.replace("#NUM#", "{:04d}".format(epoch))
            ensure_dir_exists(model_savename)
            torch.save({
                'epoch': epoch,
                'arch': 'x64',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, model_savename)

        return output

    accuracy1_values = list(map(lambda d: d.cpu().item(), accuracy1_values))
    save_data_to_csv(accuracy1_values, os.path.join(out_path, "data", os.path.basename(model_savename).replace("checkpoint_", "accuracy_").replace(".pth.tar", ".csv")), "accuracy")
    save_data_to_csv(epoch_loss_values, os.path.join(out_path, "data", os.path.basename(model_savename).replace("checkpoint_", "loss_").replace(".pth.tar", ".csv")), "loss")

def wrap_data(train_data, val_data, batch_size, batch_slide_num, batch_inst_num, workers, is_profiling, is_conditional, is_distributed):
    print('Creating dataset')
    grayer = transforms.Grayscale(num_output_channels=3)
    cropper = transforms.RandomResizedCrop(299, scale=(0.2, 1.))
    jitterer = transforms.ColorJitter(brightness=4, contrast=0.4, saturation=0.4, hue=0.01)


    def range_func(x, y):
        #return y
        return Range(x, methods="__call__")(y) if is_profiling else y
    transformations = mt.Compose(
        [
            range_func("LoadImage", mt.LoadImaged(["q", "k"], image_only=True)),
            range_func("EnsureChannelFirst", mt.EnsureChannelFirstd(["q", "k"])),
            range_func("Crop", mt.Lambdad(["q", "k"], cropper)),
            range_func("ColorJitter", mt.RandLambdad(["q", "k"], jitterer, prob=0.8)),
            range_func("Grayscale", mt.RandLambdad(["q", "k"], grayer, prob=0.2)),
            range_func("Flip0", mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=0)),
            range_func("Flip1", mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=1)),
            range_func("ToTensor", mt.ToTensord(["q", "k"], track_meta=False)),
            range_func("EnsureType", mt.EnsureTyped(["q", "k"], track_meta=False)),
            # TODO: test this
            #range_func("ToDeviced", mt.ToDeviced(["q", "k"], device="cuda:0")),
        ]
    )

    val_transformations = mt.Compose(
        [
            mt.LoadImaged(["q", "k"], image_only=True),
            mt.EnsureChannelFirstd(["q", "k"]),
            mt.Lambdad(["q", "k"], cropper),
            mt.RandLambdad(["q", "k"], jitterer, prob=0.8),
            mt.RandLambdad(["q", "k"], grayer, prob=0.2),
            mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=0),
            mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=1),
            mt.ToTensord(["q", "k"], track_meta=False),
            # TODO: test this
            #range_func("ToDeviced", mt.ToDeviced(["q", "k"], device="cuda:0")),
        ]
    )

    #ds_train = PersistentDataset(train_data, transformations, cache_dir="/var/cache/monai")
    #ds_train = CacheNTransDataset(train_data, transformations, cache_n_trans=7, cache_dir="/var/cache/monai")
    #ds_train = CacheDataset(train_data, transformations)
    ds_train = Dataset(train_data, transformations)
    #ds_val = Dataset(val_data, transformations)
    ds_val = Dataset(val_data, val_transformations)
    #ds_test = Dataset(test_data, val_transformations)

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
    else:
        train_sampler = None
    if is_conditional:
        dl_train = DataLoader(ds_train,
                            batch_sampler=samplers.MySampler(train_data, batch_size, batch_slide_num, batch_inst_num), num_workers=workers)
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=workers)

    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, shuffle=False)
    #dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=workers, shuffle=True)

    # first_sample = monai.utils.first(dl_train)
    # if first_sample is None:
    #     raise ValueError("First sample is None!")
    # for d in ["q", "k"]:
    #     print(
    #         f"[{d}] \n"
    #         f"  {d} shape: {first_sample[d].shape}\n"
    #         f"  {d} type:  {type(first_sample[d])}\n"
    #         f"  {d} dtype: {first_sample[d].dtype}"
    #     )
    #del first_sample

    print ("Number of images in DL: {}".format(len(ds_train)))
    print ("Number of batches in DL: {}".format(len(dl_train)))

    print("Dataset Created ...")
    return dl_train, dl_val, None

def main():
    args = parser.parse_args()
    reset_seeds()

    if args.batch_slide_num > args.batch_size or args.batch_inst_num > args.batch_size:
        print("Looks like you mistook m or n for n: batch_slide_num and batch_inst_num has to be less than batch_size")
        sys.exit(1)
    if args.batch_slide_num and args.batch_inst_num:
        print("Haven't implemented support for both batch_slide_num and batch_inst_num")
        sys.exit(1)
    largest_batch_sampler = max(args.batch_slide_num, args.batch_inst_num)
    if largest_batch_sampler and args.batch_size % largest_batch_sampler:
        print(f"either of n({args.batch_slide_num}),o({args.batch_inst_num}) has to evenly divide batch_size {args.batch_size}")
        sys.exit(1)
    if not args.condition:
        print("Conditional sampling false: setting batch_slide_num and batch_inst_num to 0")
        args.batch_inst_num = 0
        args.batch_slide_num = 0
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(f"root dir for MONAI is: {root_dir}")

    print('Create dataset')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    is_distributed = args.world_size > 1 or args.multiprocessing_distributed
    if is_distributed:
        torch.distributed.init_process_group(args.dist_backend)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print("=> creating model '{}'".format('x64'))
    print("condition: {}".format(args.condition))
    def do_stuff(do_checkpoint):
        reset_seeds()
        train_data, val_data, _ = build_file_list(args.data_dir, args.file_list_path)
        dl_train, dl_val, dl_test = wrap_data(train_data, val_data, args.batch_size, args.batch_slide_num,
                                              args.batch_inst_num, args.workers, args.is_profiling, args.condition,
                                              is_distributed)
        print("Dataset Created ...")

        model = condssl.builder.MoCo(InceptionV4, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, condition=args.condition, do_checkpoint=do_checkpoint)
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        print('Model builder done')
        print("Batch size: {}".format(args.batch_size))

        if args.resume:
            print ("Loading checkpoint. Make sure start_epoch is set correctly")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        model_name = condssl.builder.MoCo.__name__
        data_dir_name = list(filter(None, args.data_dir.split(os.sep)))[-1]
        out_path = os.path.join(args.out_dir, model_name, data_dir_name)
        model_filename = os.path.join(out_path, 'model', 'checkpoint_{}_{}_#NUM#_{}_m{}_n{}_o{}.pth.tar'.format(model_name, data_dir_name, args.condition, args.batch_size, args.batch_slide_num, args.batch_inst_num))

        criterion = nn.CrossEntropyLoss().cpu()
        output = train(dl_train, dl_val, model, criterion, optimizer, args.epochs, args.lr, args.cos, args.schedule, out_path, model_filename, args.is_profiling, is_distributed)
        return output
    pass_one = do_stuff(False).detach().numpy()
    pass_two = do_stuff(False).detach().numpy()
    pass_three = do_stuff(True).detach().numpy()
    assert np.allclose(pass_one, pass_two, atol=1e-6), "new_y != new_y_verified"
    assert np.allclose(pass_three, pass_two, atol=1e-6), "new_y != new_y_verified"
    pass

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    main()
