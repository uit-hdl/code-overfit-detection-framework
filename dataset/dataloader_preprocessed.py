import os
from pathlib import Path
import math
import psutil
import pickle
import ipdb
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import itertools

class PreprocessedTcgaLoader(Dataset):
    def __init__(self, dataset, batch_slide_num=4, batch_size=128, **kwargs):
        # slide_list = [ 'TCGA-21-A5DI-01A-03-TS3', ]
        self.slide2tiles = {}
        for (q, k, filename, slide_id) in dataset:
            self.slide2tiles[slide_id] = [f for f in list(os.listdir(os.path.join(self.tcga_dir, slide_id))) if "_key" not in f]
        self.idx2tiles = [os.path.join(slide_id, tile_name) for slide_id, tile_list in self.slide2tiles.items()
                          for tile_name in tile_list if '.pt' in tile_name]
        self.tiles2idx = dict(zip(self.idx2tiles, range(len(self.idx2tiles))))
        self.idx2slide = slide_list
        self.slide2idx = dict(zip(self.idx2slide, range(len(self.idx2slide))))
        self.batch_slide_num = batch_slide_num
        self.batch_size = batch_size

        # given batch size
        # impose an ordering where m // n images are sampled from each slide
        tiles_per_slide = self.batch_size // batch_slide_num
        slides_per_batch = self.batch_size // tiles_per_slide
        number_of_batches = len(self.tiles2idx) // self.batch_size
        batch_ordering = [[] for x in range(number_of_batches)]
        for i in range(number_of_batches):
            sampled_slides = 0
            for key, cand in self.slide2tiles.items():
                if sampled_slides >= slides_per_batch:
                    break
                if len(cand) > tiles_per_slide:
                    batch_ordering[i].extend([os.path.join(key, t) for t in cand[:tiles_per_slide]])
                    # doesn't work?
                    self.slide2tiles[key] = cand[tiles_per_slide:]
                    sampled_slides += 1
        self.batch_ordering = list(itertools.chain(*batch_ordering))

    def __getitem__(self, index):
        image_path = os.path.join(self.tcga_dir, self.batch_ordering[index])
        image_k_path = os.path.join(os.path.dirname(image_path), Path(os.path.basename(image_path)).resolve().stem + "_key.pt")
        image_q = torch.load(image_path)
        image_k = torch.load(image_k_path)
        return image_q, image_k

    def __len__(self):
        return len(self.batch_ordering)
