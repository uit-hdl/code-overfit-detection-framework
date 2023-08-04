import argparse
import glob
import random
import tempfile
import sys
import time
import warnings
from collections import defaultdict
from sklearn.model_selection import train_test_split

import monai.transforms as mt
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from monai.data import DataLoader, Dataset, set_track_meta, CacheDataset
from torch.utils.data import Sampler

sys.path.append("../")
import condssl.builder
import condssl.loader

from network.inception_v4 import InceptionV4
from train_util import *
import nvtx
from monai.utils.nvtx import Range
import contextlib
no_profiling = contextlib.nullcontext()

from itertools import zip_longest

#https://stackoverflow.com/a/434411
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def train(train_loader, val_loader, model, criterion, optimizer, max_epochs, lr, cos, schedule, is_profiling=True):
    epoch_loss_values = []
    metric_values = []
    epoch_times = []
    total_start = time.time()
    set_track_meta(True)

    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        train_loader_iterator = iter(train_loader)
        adjust_learning_rate(optimizer, epoch, lr, cos, schedule, max_epochs)

        # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_training_tutorial.ipynb
        # using step instead of iterate through train_loader directly to track data loading time
        # steps are 1-indexed for printing and calculation purposes
        #for i, d in enumerate(train_loader):
        for step in range(1, len(train_loader) + 1):
            step_start = time.time()
            # profiling: train dataload
            with nvtx.annotate("dataload", color="red") if is_profiling else no_profiling:
                # rng_train_dataload = nvtx.start_range(message="dataload", color="red")
                batch_data = next(train_loader_iterator)
                images_q, images_k = batch_data['q'].cuda(), batch_data['k'].cuda()
            output, target = model(im_q=images_q, im_k=images_k)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len = math.ceil(len(train_loader) / train_loader.batch_size)
            print(
                f"{step}/{epoch_len}, train_loss: {loss.item():.4f}" f" step time: {(time.time() - step_start):.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if 0 == 0:
            model_filename = os.path.join(args.out_dir, model_name, data_dir_name, "model",
                                          'checkpoint_{}_{}_{:04d}.pth.tar'.format(model_name, data_dir_name, epoch))
            torch.save({
                'epoch': epoch + 1,
                'arch': 'x64',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, model_filename)

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
parser.add_argument('--profile', default=True, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='P', help='whether to profile training or not', dest='is_profiling')
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

def find_data(src_dir, batch_size, batch_slide_num, workers, is_profiling):
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

    def range_func(x, y):
        return Range(x)(y) if is_profiling else y
    transformations = mt.Compose(
        [
            range_func("LoadImage", mt.LoadImaged(["q", "k"], image_only=True)),
            range_func("EnsureChannelFirst", mt.EnsureChannelFirstd(["q", "k"])),
            range_func("Crop", mt.Lambdad(["q", "k"], cropper)),
            range_func("ColorJitter", mt.RandLambdad(["q", "k"], jitterer, prob=0.8)),
            range_func("Grayscale", mt.RandLambdad(["q", "k"], grayer, prob=0.2)),
            range_func("Flip0", mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=0)),
            range_func("Flip1", mt.RandFlipd(["q", "k"], prob=0.5, spatial_axis=1)),
            mt.ToTensord(["q", "k"], track_meta=False),
        ]
    )

    if not all_data:
        raise RuntimeError(f"Found no data in {src_dir}")

    train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=42)
    # note that we split the train data again, not the entire dataset
    train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=42)
    ds_train, ds_val, ds_test = CacheDataset(train_data, transformations), CacheDataset(validation_data, transformations), CacheDataset(test_data, transformations)
    sampler = MySampler(ds, slide2tiles, batch_size, batch_slide_num)
    dl_train = DataLoader(ds_train, sampler=sampler, batch_size=batch_size, num_workers=workers, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=workers, shuffle=False)

    print("Dataset Created ...")
    return dl_train, dl_val, dl_test

if __name__ == '__main__':
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(f"root dir for MONAI is: {root_dir}")

    print('Create dataset')
    dl_train, dl_val, dl_test = find_data(args.data_dir, args.batch_size, args.batch_slide_num, args.workers, args.is_profiling)
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
        InceptionV4, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, condition=args.condition)
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

    criterion = nn.CrossEntropyLoss().cuda()
    train(dl_train, dl_val, model, criterion, optimizer, args.epochs, args.lr, args.cos, args.schedule)
