#!/usr/bin/env python
# coding: utf-8

'''
Train a MoCo V1 model using self-supervised learning. Stats are logged to tensorboard.
'''

import argparse
import logging
import os.path
import random
import sys
import time

import psutil

from misc.benchmark import track_method


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Moco With InceptionV4 Training')

    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
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
    parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only', dest='debug_mode')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
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
    parser.add_argument('--src-dir', default='./data/', type=str,
                        help='path to source directory')
    parser.add_argument('--out-dir', default='./out/models/', type=str,
                        help='path to output directory')
    parser.add_argument('--file-list-path', default='./out/files.csv', type=str, help='path to list of file splits')
    parser.add_argument('--batch-slide-num', default=4, type=int)
    parser.add_argument('--batch-inst-num', default=0, type=int)
    parser.add_argument('--gpu-id', nargs='+', default=[5,6], type=int, help='GPU id(s) to use.')
    parser.add_argument('--condition', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='C', help='whether to use conditional sampling or not', dest='condition')
    parser.add_argument('--checkpoint', default=True, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='X', help='whether to to sequential_checkpoint or not', dest='is_seq_ckpt')
    return parser.parse_args()

__args = parse_args()
# have to be set before importing any torch dependencies
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, __args.gpu_id))

import monai.transforms as mt
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from monai.data import DataLoader, Dataset, set_track_meta
from monai.handlers.tensorboard_handlers import SummaryWriter

sys.path.append('./')
import network.moco
import samplers

from misc.global_util import build_file_list, ensure_dir_exists

from network.inception_v4 import InceptionV4
from train_util import *


def train(train_loader, model, criterion, optimizer, max_epochs, lr, cos, schedule, out_path, model_filename, writer, device, gpu_id=None):
    if not gpu_id:
        gpu_id = [0]
    # even if you map device id 5,6 it will be seen as 0,1 in torch
    for i,g in enumerate(gpu_id):
        writer.add_scalar(f"vram_available_device_{i}", torch.cuda.get_device_properties(i).total_memory / (1024*1024))
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
            step_start = time.time()
            # profiling: train dataload
            # Download https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-3 to visualize
            batch_data = next(train_loader_iterator)
            images_q, images_k = batch_data['q'].to(device), batch_data['k'].to(device)

            output, target = model(im_q=images_q, im_k=images_k)
            target = target.to(device)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            acc5_total += acc5
            acc1_total += acc1
            iter_step = ((epoch-1)*step) + step
            writer.add_scalar("iter_loss", loss.item(), global_step=iter_step)
            writer.add_scalar("iter_acc5", acc5, global_step=iter_step)
            writer.add_scalar("iter_acc1", acc1, global_step=iter_step)
            writer.add_scalar("iter_loss", loss.item(), global_step=iter_step)
            logging.info(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f} acc1: {acc1:.2f} acc5: {acc5:.2f} step time: {(time.time() - step_start):.4f}")
        epoch_loss /= step
        acc5_total /= step
        acc1_total /= step
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

def wrap_data(train_data, val_data, batch_size, batch_slide_num, batch_inst_num, workers, is_conditional):
    logging.info('Creating dataset')
    grayer = transforms.Grayscale(num_output_channels=3)
    cropper = transforms.RandomResizedCrop(299, scale=(0.2, 1.), antialias=True)
    jitterer = transforms.ColorJitter(brightness=4, contrast=0.4, saturation=0.4, hue=0.01)

    transformations = mt.Compose(
        [
            mt.LoadImaged(["q", "k"], image_only=True),
            mt.EnsureChannelFirstd(["q", "k"]),
            mt.Lambdad(["q", "k"], cropper),
            mt.RandLambdad(["q", "k"], jitterer, prob=0.8),
            mt.RandLambdad(["q", "k"], grayer, prob=0.2),
            mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=0),
            mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=1),
            mt.ToTensord(["q", "k"], track_meta=False),
            mt.EnsureTyped(["q", "k"], track_meta=False),
        ]
    )

    val_transformations = mt.Compose(
        [
            mt.LoadImaged(["q", "k"], image_only=True),
            mt.EnsureChannelFirstd(["q", "k"]),
            mt.ToTensord(["q", "k"], track_meta=False),
        ]
    )

    ds_train = Dataset(train_data, transformations)
    ds_val = Dataset(val_data, val_transformations)

    if is_conditional:
        dl_train = DataLoader(ds_train, batch_sampler=samplers.MySampler(train_data, batch_size, batch_slide_num, batch_inst_num), num_workers=workers)
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=workers)

    if len(ds_val) > 0:
        dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, shuffle=True)
    else:
        dl_val = None

    logging.info("Dataset Created ...")
    return dl_train, dl_val

def write_args_tensorboard(args, writer):
    writer.add_text("moco_dim", str(args.moco_dim))
    writer.add_text("moco_k", str(args.moco_k))
    writer.add_text("moco_m", str(args.moco_m))
    writer.add_text("moco_t", str(args.moco_t))
    writer.add_text("mlp", str(args.mlp))
    writer.add_text("momentum", str(args.momentum))
    writer.add_text("condition", str(int(args.condition)))
    writer.add_text("lr", str(args.lr))
    writer.add_text("batch_size", str(args.batch_size))
    writer.add_text("cos", str(int(args.cos)))
    for i in args.schedule:
        writer.add_text("schedule", str(i), global_step=i)
    writer.add_text("weight_decay", str(args.weight_decay))
    writer.add_text("epochs", str(args.epochs))
    writer.add_text("workers", str(args.workers))
    writer.add_text("batch_slide_num", str(args.batch_slide_num))
    writer.add_text("batch_inst_num", str(args.batch_inst_num))


def main():
    args = parse_args()

    # if args.batch_slide_num > args.batch_size or args.batch_inst_num > args.batch_size:
    #     logging.error("Looks like you mistook m or n for n: batch_slide_num and batch_inst_num has to be less than batch_size")
    #     sys.exit(1)
    if args.condition and args.batch_slide_num and args.batch_inst_num:
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

    model_name = "moco"
    data_dir_name = list(filter(None, args.src_dir.split(os.sep)))[-1]
    out_path = os.path.join(args.out_dir, model_name, data_dir_name)
    model_filename = os.path.join(out_path, 'model',
                                  f'checkpoint_{model_name}_{data_dir_name}_#NUM#_{args.condition}_m{args.batch_size}_n{args.batch_slide_num}_o{args.batch_inst_num}_K{args.moco_k}.pth.tar')
    logging.info("Model dest filename: %s" % model_filename)

    writer = SummaryWriter(log_dir=model_filename.replace("_#NUM#", "") + "_runs")
    writer.add_text("git_sha", os.popen('git rev-parse HEAD').read().strip())

    train_data, val_data, _ = build_file_list(args.src_dir, args.file_list_path)
    if args.debug_mode:
        logging.warning("Debug mode: using only 2 * batch_size samples, 5 epochs and batch-size = 16")
        train_data = train_data[:2 * args.batch_size]
        val_data = val_data[:2 * args.batch_size]
        args.batch_size = 16
        args.epochs = 3

    write_args_tensorboard(args, writer)

    dl_train, dl_val = wrap_data(train_data, val_data, args.batch_size, args.batch_slide_num, args.batch_inst_num, args.workers, args.condition)
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
    # TODO: re-enable seq.ckpt
    model = network.moco.MoCo(
        base_encoder=InceptionV4, dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp, condition=args.condition, do_checkpoint=args.is_seq_ckpt)

    is_distributed = len(args.gpu_id) > 1
    if is_distributed:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.parallel.DataParallel(model)
    else:
        # since we set CUDA_VISIBLE_DEVICES, 0 will always be the one we want
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    writer.add_text("optimizer", optimizer.__str__())
    writer.add_text("model", str(model.__class__))
    writer.add_text("sequential_checkpointing", str(args.is_seq_ckpt))
    logging.info('Model builder done, placed on cuda()')

    criterion = nn.CrossEntropyLoss().to(device)
    writer.add_text("criterion", criterion.__str__())
    track_method(train, "train", writer, gpu_id=args.gpu_id)(dl_train, model, criterion, optimizer, args.epochs, args.lr, args.cos, args.schedule, out_path, model_filename, writer, device, gpu_id=args.gpu_id)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
