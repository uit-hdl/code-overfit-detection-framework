import random
from collections import defaultdict

from torch.utils.data import Sampler, DistributedSampler

from train_util import *
from monai.utils import Range
import contextlib
no_profiling = contextlib.nullcontext()

class MySampler(Sampler):
    """
    Conditional sampler that will generate batches made out of `batch_size` with at least `batch_slide_num` tiles from each slide
    E.g. if `batch_slide_num` is 4 and `batch_size is 32, there will be 8 slides with 4 tiles each in one batch

    Note: Don't pass in a MONAI dataset object here: the default iterator performs transformations right away
    this is very slow, and we only need the filenames to generate the indices
    """

    def __init__(self, data_source, batch_size, batch_slide_num, batch_inst_num, is_profiling=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.batch_slide_num = batch_slide_num
        self.batch_inst_num = batch_inst_num
        self.is_profiling = is_profiling

    def __iter__(self):
        with Range("MySampler") if self.is_profiling else no_profiling:
            tile_chunks = []

            slide2tiles = defaultdict(list)
            for i, d in enumerate(self.data_source):
                filename = d['filename']
                slide_id = os.path.basename(filename.split(os.sep)[-2])
                slide2tiles[slide_id].append(i)

            inst2tiles = defaultdict(list)
            for i, d in enumerate(self.data_source):
                filename = d['filename']
                institution_id = os.path.basename(filename.split(os.sep)[-2]).split("-")[1]
                inst2tiles[institution_id].append(i)

            cut_off_indices = []

            if self.batch_inst_num:
                for slide, tiles in inst2tiles.items():
                    indices = [tiles[i:i + self.batch_inst_num] for i in range(0, len(tiles), self.batch_inst_num)]
                    # Prune if a chunk created above is less than `batch_slide_num`
                    if len(indices[-1]) < self.batch_inst_num:
                        cut_off_indices.append(indices[-1])
                        indices = indices[:-1]
                    if indices:
                        random.shuffle(indices)
                        tile_chunks.append(indices)
            else:
                # generate random numbers from 0 to n in batches
                random_numbers = random.sample(range(len(self.data_source)), len(self.data_source))
                indices = [range(i, i + self.batch_size) for i in range(0, random_numbers)]
                random.shuffle(indices)
                tile_chunks = indices

            # Generate chunks of indices. If one slide has indices slide2tiles['slide'] = [1 .. 10] the result can be like
            # tile_chunks = [[1 .. 4], [ 1 .. 8 ]]
            # cut_off_indices = [ 9, 10 ]
            if self.batch_slide_num:
                for slide, tiles in slide2tiles.items():
                    indices = [tiles[i:i + self.batch_slide_num] for i in range(0, len(tiles), self.batch_slide_num)]
                    # Prune if a chunk created above is less than `batch_slide_num`
                    if len(indices[-1]) < self.batch_slide_num:
                        cut_off_indices.append(indices[-1])
                        indices = indices[:-1]
                    if indices:
                        random.shuffle(indices)
                        tile_chunks.append(indices)

            random.shuffle(tile_chunks)

            # flatten the list so we can shuffle around chunks
            tile_chunks = [item for sublist in tile_chunks for item in sublist]
            random.shuffle(tile_chunks)

            chunks_per_batch = self.batch_size // self.batch_slide_num
            ret = []
            for chunk in [tile_chunks[i:i + chunks_per_batch] for i in range(0, len(tile_chunks), chunks_per_batch)]:
                ret.append([item for sublist in chunk for item in sublist])
            return iter(ret)

    def __len__(self):
        return len(self.data_source) // (self.batch_size * self.batch_slide_num)

