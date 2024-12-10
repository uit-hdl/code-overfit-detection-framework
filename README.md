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

The UMAPs and linear probing from the paper are [here](http://github.com/uit-hdl/...). It's kept separate to make it easier to use as standalone tools.

# Dataset
The dataset is TCGA-LUSC and CPTAC. Both can be downloaded from official portals (I'm not giving a link since it keeps changing). The annotations I used are in the [annotations folder](./annotations/).
Note that for clinical annotations I used the work from [liu et al. (2018)](https://pubmed.ncbi.nlm.nih.gov/29625055/)
The annotations are downloaded from the same datasets, look for "clinical" and "slide" which should give you two separate .tsv files.
Otherwise, a lot of the information is in the filename itself.

# Installation
```bash
pip install -e .
```

# License
This code is under the Apache 2.0 license. See [LICENSE](LICENSE).