#!/usr/bin/env python

# usage: point to directory containing directories of .svs files. Or modify the script to your needs.

import openslide
import os
import glob
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np

def get_thumbnail(slide_path, output_path, width=1200, height=630):
    slide = openslide.open_slide(slide_path)
    thumbnail = slide.get_thumbnail((width, height))
    dst_path = os.path.join(output_path, os.path.basename(slide_path.split(".")[0] + ".png"))
    thumbnail.save(dst_path)
    print("Saved thumbnail %s" % dst_path)
    slide.close()

if __name__ == "__main__":
    print("Generating thumbnails...")
    slide_path = sys.argv[1]
    output_path = sys.argv[2]
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for i, directory in enumerate(glob.glob(f"{slide_path}{os.sep}*")):
        for filename in glob.glob(f"{directory}{os.sep}**{os.sep}*.svs", recursive=True):
            if os.path.isfile(filename):
                get_thumbnail(filename, output_path)
    print("Done!")
