import argparse
import glob
import os
import pathlib
from pathlib import Path

from monai.data import partition_dataset, partition_dataset_classes
import monai.transforms as mt
import torch.utils.data
import torchvision.transforms as transforms
from monai.data import DataLoader, Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--src-dir', default='./data/', type=str, dest='src_dir',
                    help='path to output directory')
parser.add_argument('--dst-dir', default='./data/augmented/', type=str, dest='out_dir',
                    help='path to output directory')
args = parser.parse_args()

print("=> creating model '{}'".format('x64'))

grayer = transforms.Grayscale(num_output_channels=1)
cropper = transforms.RandomResizedCrop(299, scale=(0.2, 1.))
jitterer = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

all_data = []

for directory in glob.glob(f"{args.src_dir}{os.sep}*"):
    for file in glob.glob(f"{directory}{os.sep}**{os.sep}*", recursive=True):
        if os.path.isfile(file):
            all_data.append({"q": file, "k": file, "filename": file, "label": os.path.basename(os.path.dirname(file))})

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

print('Creating dataset')
ds = Dataset(all_data, transformations)
dl = DataLoader(ds)
print("Dataset Created ...")

for data in tqdm(dl):
    q_list, k_list, filenames = data["q"], data["k"], data["filename"]
    for q, k, filename in zip(q_list, k_list, filenames):
        tile_name = Path(os.path.basename(filename)).resolve().stem
        slide_name = pathlib.PurePath(filename).parent.name
        slide_folder = os.path.join(args.out_dir, slide_name)
        if not os.path.exists(slide_folder):
            Path(slide_folder).mkdir(parents=True, exist_ok=False)
        torch.save(q, os.path.join(slide_folder, tile_name + ".pt"))
        torch.save(k, os.path.join(slide_folder, tile_name + "_key.pt"))
