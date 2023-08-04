import argparse
import glob
import numpy as np
import random
from collections import defaultdict
import sys
import time
import warnings
import random
import itertools


import monai.transforms as mt
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from monai.data import DataLoader, Dataset
from torch.utils.data import Sampler

sys.path.append("../")
import condssl.builder
import condssl.loader
from dataset.dataloader import TCGA_CPTAC_Dataset

from dataset.dataloader_preprocessed import PreprocessedTcgaLoader
from network.inception_v4 import InceptionV4
from train_util import *

from itertools import zip_longest

#https://stackoverflow.com/a/434411
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    start_time = time.time()
    # for i, ((images, transformed_images)) in enumerate(train_loader):
    for i, d in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # for 2 lines below:
        #with torch.autocast(device_type='cuda', dtype=torch.float16):
        images_q, images_k = d['q'], d['k']
        output, target = model(im_q=images_q.cuda(), im_k=images_k.cuda())
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images_q.size(0))
        top1.update(acc1[0], images_q.size(0))
        top5.update(acc5[0], images_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            end_time = time.time()
            print("Time elapsed: {}".format(end_time - start_time))
            start = time.time()
    
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

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
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--data_dir', default='./data/', type=str,
                    help='path to output directory')
parser.add_argument('--out_dir', default='./models/', type=str,
                    help='path to output directory')
parser.add_argument('--batch_slide_num', default=4, type=int)
parser.add_argument('--condition', default=True, type=bool)

encoder = InceptionV4

criterion = nn.CrossEntropyLoss().cuda()


def find_data(src_dir, batch_size, batch_slide_num, workers):
    def my_collate_fn(data, *args, **kwargs):
        pass


    class MySampler(Sampler):
        def __init__(self, data_source, slide2tiles, batch_size, batch_slide_num):
            self.data_source = data_source
            self.batch_size = batch_size
            self.batch_slide_num = batch_slide_num
            self.slide2tiles = slide2tiles

        def __iter__(self):
            slides_per_batch = self.batch_size // self.batch_slide_num
            tile_chunks = []
            for slide,tiles in slide2tiles.items():
                indices = [tiles[i:i + self.batch_slide_num] for i in range(0, len(tiles), self.batch_slide_num)]
                if len(indices[-1]) < self.batch_slide_num:
                    indices = indices[:-1]
                random.shuffle(indices)
                if indices:
                    tile_chunks.append(indices)

            random.seed(42)
            random.shuffle(tile_chunks)

            # flatten the list
            tile_chunks = [item for sublist in tile_chunks for item in sublist]
            random.shuffle(tile_chunks)
            # flatten the list MORE
            tile_chunks = [item for sublist in tile_chunks for item in sublist]

            for i in tile_chunks:
                yield i

        def __len__(self):
            return len(self.data_source) // (self.batch_size * self.batch_slide_num)

    print('Creating dataset')
    grayer = transforms.Grayscale(num_output_channels=3)
    cropper = transforms.RandomResizedCrop(299, scale=(0.2, 1.))
    jitterer = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    all_data = []
    slide2tiles = defaultdict(list)
    i = 0
    for directory in glob.glob(f"{src_dir}{os.sep}*"):
        for file in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
            if os.path.isfile(file):
                slide_id = os.path.basename(os.path.dirname(file))
                slide2tiles[slide_id].append(i)
                i += 1
                all_data.append({"q": file, "k": file, "filename": file})

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
        ]
    )

    if not all_data:
        raise RuntimeError(f"Found no data in {src_dir}")

    ds = Dataset(all_data, transformations)
    dl = DataLoader(ds, sampler=MySampler(ds, slide2tiles, batch_size, batch_slide_num), batch_size=batch_size, num_workers=workers, shuffle=False)
    #dl = TCGA_CPTAC_Dataset(all_data, batch_size=batch_size, num_workers=workers, shuffle=True)

    print("Dataset Created ...")
    for x in dl:
        pass
    return ds, dl


if __name__ == '__main__':
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)

    print('Create dataset')
    ds, dl = find_data(args.data_dir, args.batch_size, args.batch_slide_num, args.workers)
    print("Dataset Created ...")

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
    model = condssl.builder.MoCo(
        encoder, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, condition=args.condition)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    print('Model builder done, placed on cuda()')
    print("Batch size: {}".format(args.batch_size))

    if args.resume:
        print ("Loading checkpoint. Make sure start_epoch is set correctly")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model_name = condssl.builder.MoCo.__name__
    data_dir_name = args.data_dir.split(os.sep)[-1]
    ensure_dir_exists(os.path.join(args.out_dir, model_name, data_dir_name, "model"))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.cos, args.schedule, args.epochs)

        # train for one epoch
        train(dl, model, criterion, optimizer, epoch, args)
        if 0 == 0:
            model_filename = os.path.join(args.out_dir, model_name, data_dir_name, "model",
                                          'checkpoint_{}_{}_{:04d}.pth.tar'.format(model_name, data_dir_name, epoch))
            torch.save({
                'epoch': epoch + 1,
                'arch': 'x64',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, model_filename)
