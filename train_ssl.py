import ipdb
from torchvision.models.maxvit import WindowDepartition
from train_util import *
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from GPUtil import showUtilization as gpu_usage

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import condssl.loader
import condssl.builder

import sys
from network.inception_v4 import InceptionV4
from dataset.dataloader import TCGA_CPTAC_Dataset

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def collate_fn_moco(batch):
    q_list = []
    k_list = []
    for imgs in batch:
        for (img, transformed) in imgs:
            q_list.append(img.unsqueeze(0))
            k_list.append(transformed.unsqueeze(0))
    import ipdb; ipdb.set_trace()
    return torch.cat(q_list, dim=0), torch.cat(k_list, dim=0)


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
    # for i, ((images, transformed_images)) in enumerate(train_loader):
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # for 2 lines below:
        # with torch.autocast(device_type='cuda', dtype=torch.float32):

        output, target = model(im_q=images[0].cuda(), im_k=images[1].cuda())
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


    
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

parser.add_argument('--partition_name', default='train_Lung', type=str)
parser.add_argument('--data_dir', default='./data/', type=str,
                    help='path to output directory')
parser.add_argument('--split_dir', default='./split/', type=str,
                    help='path to output directory')
parser.add_argument('--out_dir', default='./models/', type=str,
                    help='path to output directory')
parser.add_argument('--batch_slide_num', default=4, type=int)
parser.add_argument('--condition', default=True, type=bool)

args = parser.parse_args()

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

encoder = InceptionV4

model = condssl.builder.MoCo(
    encoder, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, condition=args.condition)

model = model.cuda()

print('Model builder Done.')
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

augmentation = [
        transforms.RandomResizedCrop(299, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([condssl.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
        # normalize
    ]

print('Create dataset')


print("Batch size: {}".format(args.batch_size))
train_dataset = TCGA_CPTAC_Dataset(cptac_dir=args.data_dir + "/CPTAC/tiles/",
                          tcga_dir=os.path.join(args.data_dir, "TCGA", "tiles"),
                          split_dir=args.split_dir,
                          transform=TwoCropsTransform(transforms.Compose(augmentation)),
                          # TODO: why isn't batch_size default here?
                          batch_size=args.batch_size,
                          batch_slide_num=args.batch_slide_num)
#train_dataset = datasets.ImageFolder(args.data_dir + "/TCGA/tiles/", TwoCropsTransform(transforms.Compose(augmentation)))


print("Dataset Created ...")

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=12,
        pin_memory=False, drop_last=True)
        #, collate_fn=collate_fn_moco)

if args.resume:
    print ("Loading checkpoint. Make sure start_epoch is set correctly")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


for epoch in range(args.start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, args)

    if 0 == 0:
    #TODO: fixup
    # if (epoch + 1) % 25 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'x64',
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.out_dir, epoch))
