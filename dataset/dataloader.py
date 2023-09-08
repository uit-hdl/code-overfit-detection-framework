import itertools
import math
import os
from collections import defaultdict

import cv2
import numpy as np
import psutil
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from itertools import zip_longest

#https://stackoverflow.com/a/434411
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class MyDataLoader(DataLoader):
    def __init__(self, ds, batch_slide_num=4, **kwargs):
        # slide_list = [ 'TCGA-21-A5DI-01A-03-TS3', ]
        slide2tiles = defaultdict(list)
        for (q, k, tile_name, slide_id) in dataset_list:
            self.slide2tiles[slide_id].append((q, k, tile_name))
        self.idx2tiles = [os.path.join(slide_id, tile_name) for slide_id, tile_list in self.slide2tiles.items()
                          for tile_name in tile_list if 'jpg' in tile_name]
        self.tiles2idx = dict(zip(self.idx2tiles, range(len(self.idx2tiles))))
        self.idx2slide = list(self.slide2tiles.keys())
        self.slide2idx = dict(zip(self.idx2slide, range(len(self.idx2slide))))
        self.transform = transform
        self.batch_slide_num = batch_slide_num
        self.batch_size = batch_size

        # given batch size
        # impose an ordering where m // n images are sampled from each slide
        tiles_per_slide = self.batch_size // batch_slide_num
        slides_per_batch = self.batch_size // tiles_per_slide

        grouper(slide2tiles, tiles_per_slide)

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
        self.tile_list = [None] * len(self.batch_ordering)

        for i, tile_name in enumerate(self.batch_ordering):
            #image = cv2.imread(os.path.join(self.tcga_dir, tile_name))
            img_arr = Image.open(os.path.join(self.tcga_dir, tile_name))
            self.tile_list[i] = img_arr
            if i % 1000 == 0:
                print("Iterating images: {}/{}".format(i, len(self.batch_ordering)))
                print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

    def __getitem__(self, index):
        image = self.tile_list[index]
        return self.transform(image)
        #
        #return self.batch_ordering[index]
        #
        # tile_name = self.batch_ordering[index]
        # if "TCGA" in tile_name:
        #     image = cv2.imread(os.path.join(self.tcga_dir, tile_name))
        # else:
        #     image = cv2.imread(os.path.join(self.cptac_dir, tile_name))
        #
        # image = Image.fromarray(image)
        # (img_tensor, transformed_image_tensor) = self.transform(image)
        # return img_tensor, transformed_image_tensor

    def __len__(self):
        return len(self.batch_ordering)

class TCGA_CPTAC_Dataset(Dataset):
    def __init__(self, dataset_list, transform=None, batch_slide_num=4, batch_size=128, **kwargs):
        # slide_list = [ 'TCGA-21-A5DI-01A-03-TS3', ]
        slide2tiles = defaultdict(list)
        for (q, k, tile_name, slide_id) in dataset_list:
            self.slide2tiles[slide_id].append(tile_name)
        self.idx2tiles = [os.path.join(slide_id, tile_name) for slide_id, tile_list in self.slide2tiles.items()
                          for tile_name in tile_list if 'jpg' in tile_name]
        self.tiles2idx = dict(zip(self.idx2tiles, range(len(self.idx2tiles))))
        self.idx2slide = list(self.slide2tiles.keys())
        self.slide2idx = dict(zip(self.idx2slide, range(len(self.idx2slide))))
        self.transform = transform
        self.batch_slide_num = batch_slide_num
        self.batch_size = batch_size

        # given batch size
        # impose an ordering where m // n images are sampled from each slide
        tiles_per_slide = self.batch_size // batch_slide_num
        slides_per_batch = self.batch_size // tiles_per_slide

        grouper(slide2tiles, tiles_per_slide)

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
        self.tile_list = [None] * len(self.batch_ordering)

        for i, tile_name in enumerate(self.batch_ordering):
            #image = cv2.imread(os.path.join(self.tcga_dir, tile_name))
            img_arr = Image.open(os.path.join(self.tcga_dir, tile_name))
            self.tile_list[i] = img_arr
            if i % 1000 == 0:
                print("Iterating images: {}/{}".format(i, len(self.batch_ordering)))
                print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

    def __getitem__(self, index):
        image = self.tile_list[index]
        return self.transform(image)
        #
        #return self.batch_ordering[index]
        #
        # tile_name = self.batch_ordering[index]
        # if "TCGA" in tile_name:
        #     image = cv2.imread(os.path.join(self.tcga_dir, tile_name))
        # else:
        #     image = cv2.imread(os.path.join(self.cptac_dir, tile_name))
        #
        # image = Image.fromarray(image)
        # (img_tensor, transformed_image_tensor) = self.transform(image)
        # return img_tensor, transformed_image_tensor

    def __len__(self):
        return len(self.batch_ordering)


class TCGA_CPTAC_Bag_Dataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        # slide_list = pickle.load(open(os.path.join(split_dir, 'case_split_2yr.pkl'), 'rb'))[mode + '_id']
        # slide_list = [
        #         'TCGA-39-5016-01A-01-BS1',
        # ]
        slide_list = os.popen("find {}/TCGA/tiles/ -maxdepth 1 -type d ".format(data_dir)).read().strip('\n').split('\n')[1:]
        slide_list = list(map(lambda s: s.rsplit('/', 1)[1].split('.')[0], slide_list))
        keep_train = math.floor(len(slide_list) * 0.7)
        keep_val = keep_train + math.floor(len(slide_list) * 0.1)
        #keep_test = keep_val + math.floor(len(slide_list) * 0.2)
        if mode == 'train':
            slide_list = slide_list[:keep_train]
        elif mode == 'val':
            slide_list = slide_list[keep_train:keep_val]
        elif mode == 'test':
            slide_list = slide_list[keep_val:]
        self.slide2tiles = {}
        for slide_id in slide_list:
            if "TCGA" in slide_id:
                tile_dir = self.data_dir + '/TCGA/tiles/'
            else:
                tile_dir = self.data_dir + '/CPTAC/tiles/'
            self.slide2tiles[slide_id] = os.listdir(os.path.join(tile_dir, slide_id))
        self.idx2tiles = [os.path.join(slide_id, tile_name) for slide_id, tile_list in self.slide2tiles.items() 
                     for tile_name in tile_list if 'jpg' in tile_name]
        self.tiles2idx = dict(zip(self.idx2tiles, range(len(self.idx2tiles))))
        self.idx2slide = slide_list
        self.slide2idx = dict(zip(self.idx2slide, range(len(self.idx2slide))))
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        tile_path = self.idx2tiles[index]
        slide_id, tile_name = tile_path.split('/')
        tile_idx = self.tiles2idx[tile_path]
        slide_idx = self.slide2idx[slide_id]
        if "TCGA" in tile_path:
            prefix = self.data_dir + '/TCGA/tiles/'
        else:
            prefix = self.data_dir + '/CPTAC/tiles/'
        image = cv2.imread(os.path.join(prefix, tile_path))
        
        if image is None:
            print (os.path.join(prefix, tile_path))
            print (os.path.join(prefix, tile_path))
            print (os.path.join(prefix, tile_path))
            print (os.path.join(prefix, tile_path))
        image = Image.fromarray(image)
        image_tensor = self.transform(image)
        return image_tensor, tile_idx, slide_idx
    
    def __len__(self):
        return len(self.idx2tiles)

