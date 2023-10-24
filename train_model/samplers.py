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

    def _sample_batch_from_group(self, confound2tiles, batch_size):
        # Generate chunks of indices. If one slide has indices slide2tiles['slide'] = [1 .. 10] the result can be like
        # tile_chunks = [[1 .. 4], [ 1 .. 8 ]]
        # cut_off_indices = [ 9, 10 ]
        tile_chunks = []
        cut_off_indices = []
        for confounder, tiles in confound2tiles.items():
            random.shuffle(tiles)
            indices = [tiles[i:i + batch_size] for i in range(0, len(tiles), batch_size)]
            # Prune if a chunk created above is less than `batch_slide_num`
            if len(indices[-1]) < batch_size:
                cut_off_indices.append({confounder: indices[-1]})
                indices = indices[:-1]
            if indices:
                random.shuffle(indices)
                tile_chunks.append({confounder: indices})
        return tile_chunks, cut_off_indices

    def __iter__(self):
        with Range("MySampler") if self.is_profiling else no_profiling:
            tile_chunks = []

            slide2tiles = defaultdict(list)
            inst2tiles = defaultdict(list)
            for i, d in enumerate(self.data_source):
                filename = d['filename']
                slide_id = os.path.basename(filename.split(os.sep)[-2])
                institution_id = slide_id.split("-")[1]

                slide2tiles[slide_id].append(i)
                inst2tiles[institution_id].append(i)

            #cut_off_indices = []

            if self.batch_inst_num:
                tile_chunks_inst, inst_cutoff = self._sample_batch_from_group(inst2tiles, self.batch_inst_num)
            if self.batch_slide_num:
                tile_chunks_slide, slide_cutoff = self._sample_batch_from_group(slide2tiles, self.batch_slide_num)

            if self.batch_inst_num and self.batch_slide_num:
                raise NotImplementedError("Not implemented")
                # tile_chunks = []
                # if self.batch_inst_num > self.batch_slide_num:
                #     d = tile_chunks_slide[0]
                #     slide, chunks = list(d.keys())[0], list(d.values())[0]
                #     institution_id = slide.split("-")[1]
                #     tile_chunks = tile_chunks_slide[0] + tile_chunks_inst[0:(self.batch_inst_num - self.batch_slide_num)]
                # pass

            elif self.batch_slide_num and not self.batch_inst_num:
                tile_chunks = tile_chunks_slide
            elif self.batch_inst_num and not self.batch_slide_num:
                tile_chunks = tile_chunks_inst

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

