# Code: Open-source framework for detecting bias and overfitting for large pathology images
this readme contains code used in the paper "Open-source framework for detecting bias and overfitting for large pathology images".

The code here covers:
* [preprocessing slides into tiles using Vahadane normalization](./preprocessing/process_tcga.py)
* [creating a SSL models based on MoCo v1](./train_mnodel/train_ssl.py)
  * this model can also be configured to do _conditional sampling_
* [exporting embeddings to zarr arrays](./feature_extraction/)
* [generating a tile-level annotation file](./preprocessing/gen_tcga_tile_labels.py)
* [fine-tuning phikon/MoCo v1](./train_model/relabelling.py)
* [Creating figures for the paper](./figures/)

The code for UMAPs and linear probing from the paper are @ [github.com/uit-hdl/feature-inspect](http://github.com/uit-hdl/feature-inspect). It's kept separate to make it easier to use as standalone tools.

# Dataset
The dataset is TCGA-LUSC. It can be downloaded from official portals (I'm not giving a link since it keeps changing). The annotations I used are in the [annotations folder](./annotations/).
For clinical annotations, you only need to use the filename, but if you want extended clinical information I recommend download TCGA annotations here from [liu et al. (2018)](https://pubmed.ncbi.nlm.nih.gov/29625055/)
The annotations are downloaded from the same datasets, look for "clinical" and "slide" which should give you two separate .tsv files.

# Installation
```bash
pip install -r requirements.txt
```

# Recreating the paper
Assuming you have the raw dataset from the TCGA portal in "/data/TCGA-LUSC". We use ipython since regular python may give a "module not found":
```bash
# create tiles and annotations
# the current default color normalization is Vahadane, but you can change this in the script
# according to many other papers, normalization has little impact on TSS bias, so you could consider changing color normalization to speed it up
ipython preprocessing/process_tcga.py -- --wsi-path /data/TCGA-LUSC --out-dir /data/TCGA-LUSC-tiles
ipython preprocessing/gen_tcga_tile_labels.py -- --data-dir /data/TCGA-LUSC-tiles --out-dir out

# train the model. Can be skipped if you just want to use PhikonV2. For our setup the training took about 3 days per model
# you can also skip this and just use PhikonV2 (next steps) to avoid training
ipython train_model/train_ssl.py -- --condition --batch-slide-num 4 --src-dir /data/TCGA-LUSC-tiles --epochs 300 --moco-k 128
ipython train_model/train_ssl.py -- --no-condition --src-dir /data/TCGA-LUSC-tiles --epochs 300 --moco-k 128
ipython train_model/train_ssl.py -- --no-condition --src-dir /data/TCGA-LUSC-tiles --epochs 300 --moco-k 65536

ipython feature_extraction/extract_features_phikon2.py -- --src-dir /data/TCGA-LUSC-tiles --out-dir out/
ipython feature_extraction/extract_features_inceptionv4.py -- --src-dir /data/TCGA-LUSC-tiles \
  --out-dir out --model-pth out/ --model-pth 'out/models/MoCo/TCGA_LUSC/model/checkpoint_MoCo_TCGA_LUSC_0200_False_m128_n0_o0_K128.pth.tar'
# ..repeat for other models..
```
After running the above, you'll have embeddings saved in `./out/*.zarr`. These can then be used by our `feature_inspect` package.
To view model stats for InceptionV4, you can use tensorboard: `tensorboard --logdir=out/models/MoCo/TCGA_LUSC/model/`

# License
This code is under the Apache 2.0 license. See [LICENSE](LICENSE).
