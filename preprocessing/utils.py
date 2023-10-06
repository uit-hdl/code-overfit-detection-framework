import cv2
import sys
import os
from openslide import OpenSlide
from pathlib import Path
from PIL import Image
import numpy as np
import staintools


def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag


def wsi_to_tiles(slide_name, slide_dest, refer_img, s):
    normalizer = staintools.StainNormalizer(method='vahadane')
    refer_img = staintools.read_image(refer_img)
    normalizer.fit(refer_img)
    count = 0
    sys.stdout.write('Start task: %s->%s \n' % (slide_name, slide_dest))
    slide_id = slide_dest.rsplit(os.sep, 1)[1].split('.')[0]
    img = OpenSlide(slide_name)
    if str(img.properties.values.__self__.get('tiff.ImageDescription')).split("|")[1] == "AppMag = 40":
        sz = 2048
        seq = 1536
    else:
        sz = 1024
        seq = 768
    [w, h] = img.dimensions
    for x in range(1, w, seq):
        for y in range(1, h, seq):
            img_path = os.path.join(slide_dest, str(x) + "_" + str(y) + '.jpg')
            if os.path.exists(img_path):
                print ("skipping %s - already done" % img_path)
                continue
            print("writing %s" % img_path)
            img_tmp = img.read_region(location=(x, y), level=0, size=(sz, sz)) \
                            .convert("RGB").resize((299, 299), Image.ANTIALIAS)
            grad = getGradientMagnitude(np.array(img_tmp))
            unique, counts = np.unique(grad, return_counts=True)
            if counts[np.argwhere(unique <= 15)].sum() < 299 * 299 * s:
                img_tmp = normalizer.transform(np.array(img_tmp))
                img_tmp = Image.fromarray(img_tmp)
                img_tmp.save(img_path, 'JPEG', optimize=True, quality=94)
                count += 1
    sys.stdout.write('End task %s->%s with %d tiles\n' % (slide_name,slide_dest,count))

def ensure_dir_exists(path):
    dest_dir = os.path.dirname(path)
    if not os.path.exists(dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        print(f"mkdir: '{dest_dir}'")

