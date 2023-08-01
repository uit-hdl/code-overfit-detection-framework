import argparse
import os
import pathlib
from pathlib import Path

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

import condssl.builder
import condssl.loader


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', default='./data/', type=str,
                    help='path to output directory')
parser.add_argument('--out_dir', default='./models/', type=str,
                    help='path to output directory')
args = parser.parse_args()

print("=> creating model '{}'".format('x64'))

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


train_dataset = datasets.ImageFolder(args.data_dir + "/TCGA/tiles/", TwoCropsTransform(transforms.Compose(augmentation)))
print("Dataset Created ...")

number_of_images = len(train_dataset)

for i, ((images_q, images_k), _) in tqdm(enumerate(train_dataset)):
    filename, _ = train_dataset.imgs[i]
    tile_name = Path(os.path.basename(filename)).resolve().stem
    slide_name = pathlib.PurePath(filename).parent.name
    slide_folder = os.path.join(args.out_dir, slide_name)
    if not os.path.exists(slide_folder):
        os.mkdir(slide_folder)
    torch.save(images_q, os.path.join(slide_folder, tile_name + ".pt"))
    torch.save(images_k, os.path.join(slide_folder, tile_name + "_key.pt"))
