import argparse
import csv
import os.path
import random
import sys
import psutil
import tempfile
import time
import warnings
import logging

import monai.transforms as mt
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from monai.data import DataLoader, Dataset, set_track_meta
from monai.handlers.tensorboard_handlers import SummaryWriter
from torch.utils.data import Sampler

sys.path.append('./')
import condssl.builder
import condssl.loader
import samplers

from misc.global_util import build_file_list, ensure_dir_exists

from network.inception_v4 import InceptionV4
from train_util import *
from monai.utils import Range
import contextlib
no_profiling = contextlib.nullcontext()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
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
parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only', dest='debug_mode')
parser.add_argument('--profile', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='P', help='whether to profile training or not', dest='is_profiling')
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
                    help='queue size; number of negative keys (default moco: 65536)')
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
parser.add_argument('--condition', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='C', help='whether to use conditional sampling or not', dest='condition')

def train(train_loader, model, criterion, optimizer, max_epochs, lr, cos, schedule, out_path, model_filename, writer, is_profiling, is_distributed):
    epoch_loss_values = []
    accuracy1_values = []
    accuracy5_values = []
    writer.add_scalar("vram_available_device_0", torch.cuda.get_device_properties(0).total_memory / (1024*1024))
    set_track_meta(True)
    model_savename = ""
    step = 0

    for epoch in range(1, max_epochs + 1):
        logging.info("-" * 10)
        logging.info(f"epoch {epoch}/{max_epochs}")
        model.train()
        epoch_loss = 0
        acc5_total = 0
        acc1_total = 0
        train_loader_iterator = iter(train_loader)
        adjust_learning_rate(optimizer, epoch, lr, cos, schedule, max_epochs)

        for step in range(1, len(train_loader) + 1):
            if is_distributed:
                train_loader.batch_sampler.set_epoch(epoch)
            with Range("Step"):
                step_start = time.time()
                # profiling: train dataload
                # Download https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-3 to visualize
                with Range("Dataload") if is_profiling else no_profiling:
                    batch_data = next(train_loader_iterator)
                    images_q, images_k = batch_data['q'].cuda(), batch_data['k'].cuda()
                with Range("forward") if is_profiling else no_profiling:
                    output, target = model(im_q=images_q, im_k=images_k)
                    loss = criterion(output, target)

                with Range("accuracy") if is_profiling else no_profiling:
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                with Range("backward_loss") if is_profiling else no_profiling:
                    loss.backward()
                with Range("update_optimizer") if is_profiling else no_profiling:
                    optimizer.step()

                epoch_loss += loss.item()
                acc5_total += acc5
                acc1_total += acc1
                writer.add_scalar("iter_loss", loss.item(), global_step=(epoch*step) + step)
                writer.add_scalar("iter_acc5", acc5, global_step=(epoch*step) + step)
                writer.add_scalar("iter_acc1", acc1, global_step=(epoch*step) + step)
                writer.add_scalar("iter_loss", loss.item(), global_step=(epoch*step) + step)
                logging.info(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f} acc1: {acc1:.2f} acc5: {acc5:.2f} step time: {(time.time() - step_start):.4f}")
                writer.add_scalar("ram_used_mb", psutil.virtual_memory()[3] / 1000000, global_step=(epoch*step) + step)
                writer.add_scalar("cpu_used", psutil.cpu_percent(), global_step=(epoch * step) + step)
                writer.add_scalar("vram_used_device_0", torch.cuda.memory_reserved(0), global_step=(epoch * step) + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        acc5_total /= step
        accuracy5_values.append(acc5_total)
        acc1_total /= step
        accuracy1_values.append(acc1_total)
        writer.add_scalar("loss", epoch_loss, global_step=epoch)
        writer.add_scalar("accuracy1", acc1, global_step=epoch)
        writer.add_scalar("accuracy5", acc5, global_step=epoch)
        logging.info(f"epoch {epoch} average loss: {epoch_loss:.4f}")

        if not epoch % 20 or epoch == max_epochs:
            model_savename = model_filename.replace("#NUM#", "{:04d}".format(epoch))
            ensure_dir_exists(model_savename)
            logging.info("Saved model to %s" % model_savename)
            torch.save({
                'epoch': epoch,
                'arch': 'x64',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, model_savename)

def wrap_data(train_data, val_data, batch_size, batch_slide_num, batch_inst_num, workers, is_profiling, is_conditional, is_distributed):
    logging.info('Creating dataset')
    grayer = transforms.Grayscale(num_output_channels=3)
    cropper = transforms.RandomResizedCrop(299, scale=(0.2, 1.), antialias=True)
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
        ]
    )

    ds_train = Dataset(train_data, transformations)
    ds_val = Dataset(val_data, val_transformations)

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
    else:
        train_sampler = None
    if is_conditional:
        dl_train = DataLoader(ds_train, batch_sampler=samplers.MySampler(train_data, batch_size, batch_slide_num, batch_inst_num), num_workers=workers)
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=workers)

    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, shuffle=True)

    logging.info("Dataset Created ...")
    return dl_train, dl_val

def write_args_tensorboard(args, writer):
    writer.add_scalar("moco_dim", args.moco_dim)
    writer.add_scalar("moco_k", args.moco_k)
    writer.add_scalar("moco_m", args.moco_m)
    writer.add_scalar("moco_t", args.moco_t)
    writer.add_scalar("mlp", args.mlp)
    writer.add_scalar("momentum", args.momentum)
    writer.add_scalar("condition", int(args.condition))
    writer.add_scalar("lr", args.lr)
    writer.add_scalar("batch_size", args.batch_size)
    writer.add_scalar("is_profiling", int(args.is_profiling))
    writer.add_scalar("cos", int(args.cos))
    for i in args.schedule:
        writer.add_scalar("schedule", 10, global_step=i)
    writer.add_scalar("weight_decay", args.weight_decay)
    writer.add_scalar("epochs", args.epochs)
    writer.add_scalar("workers", args.workers)
    writer.add_scalar("batch_slide_num", args.batch_slide_num)
    writer.add_scalar("batch_inst_num", args.batch_inst_num)


def main():
    args = parser.parse_args()

    if args.batch_slide_num > args.batch_size or args.batch_inst_num > args.batch_size:
        logging.error("Looks like you mistook m or n for n: batch_slide_num and batch_inst_num has to be less than batch_size")
        sys.exit(1)
    if args.batch_slide_num and args.batch_inst_num:
        logging.error("Haven't implemented support for both batch_slide_num and batch_inst_num")
        sys.exit(1)
    largest_batch_sampler = max(args.batch_slide_num, args.batch_inst_num)
    if largest_batch_sampler and args.batch_size % largest_batch_sampler:
        logging.error(f"either of n({args.batch_slide_num}),o({args.batch_inst_num}) has to evenly divide batch_size {args.batch_size}")
        sys.exit(1)
    if not args.condition:
        logging.info("Conditional sampling false: setting batch_slide_num and batch_inst_num to 0")
        args.batch_inst_num = 0
        args.batch_slide_num = 0
    if not torch.cuda.is_available():
        logging.error('No GPU device available')
        sys.exit(1)

    logging.info('Create dataset')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    is_distributed = args.world_size > 1 or args.multiprocessing_distributed
    if is_distributed:
        torch.distributed.init_process_group(args.dist_backend)

    model_name = condssl.builder.MoCo.__name__
    data_dir_name = list(filter(None, args.data_dir.split(os.sep)))[-1]
    out_path = os.path.join(args.out_dir, model_name, data_dir_name)
    model_filename = os.path.join(out_path, 'model', 'checkpoint_{}_{}_#NUM#_{}_m{}_n{}_o{}_K{}.pth.tar'.format(model_name, data_dir_name, args.condition, args.batch_size, args.batch_slide_num, args.batch_inst_num, args.moco_k))
    logging.info("Model dest filename: %s" % model_filename)

    writer = SummaryWriter(log_dir=model_filename.replace("_#NUM#", "") + "_runs")
    writer.add_scalar("is_distributed", int(not is_distributed))

    train_data, val_data, _ = build_file_list(args.data_dir, args.file_list_path)
    if args.debug_mode:
        logging.warning("Debug mode: using only 2 * batch_size samples, 5 epochs and batch-size = 16")
        train_data = train_data[:2 * args.batch_size]
        val_data = val_data[:2 * args.batch_size]
        args.batch_size = 16
        args.epochs = 3

    write_args_tensorboard(args, writer)

    dl_train, dl_val = wrap_data(train_data, val_data, args.batch_size, args.batch_slide_num, args.batch_inst_num, args.workers, args.is_profiling, args.condition, is_distributed)
    writer.add_scalar("images_in_ds_train", len(dl_train.dataset))
    writer.add_scalar("batches_in_ds_train", len(dl_train))
    dropped_off_tiles = len(dl_train.dataset) - (len(dl_train) * dl_train.batch_sampler.batch_size)
    writer.add_scalar("dropped_off_tiles_ds_train", dropped_off_tiles)

    logging.info("Dataset Created ...")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    logging.info("=> creating model '{}'".format('x64'))
    model = condssl.builder.MoCo(
        base_encoder=InceptionV4, dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp, condition=args.condition, do_checkpoint=not is_distributed)

    model = model.cuda()
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    writer.add_text("optimizer", optimizer.__str__())
    writer.add_text("model", str(model.__class__))
    logging.info('Model builder done, placed on cuda()')

    criterion = nn.CrossEntropyLoss().cuda()
    writer.add_text("criterion", criterion.__str__())
    train(dl_train, model, criterion, optimizer, args.epochs, args.lr, args.cos, args.schedule, out_path, model_filename, writer, args.is_profiling, is_distributed)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
