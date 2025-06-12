#!/usr/bin/env bash

set -xe

test -e out/files.csv && rm -v out/files.csv
ipython train_model/train_ssl.py --  --src-dir /Data/TCGA_LUSC/tiles/ --no-checkpoint --epochs 3 --moco-k 65536 --batch-size 32  --out-dir out_batch32

test -e out/files.csv && rm  -v out/files.csv
ipython train_model/train_ssl.py --  --src-dir /Data/TCGA_LUSC/tiles/ --checkpoint --epochs 3 --moco-k 65536 --batch-size 32  --out-dir out_batch32


test -e out/files.csv && rm -v out/files.csv
ipython train_model/train_ssl.py --  --src-dir /Data/TCGA_LUSC/tiles/ --no-checkpoint --epochs 3 --moco-k 65536 --batch-size 64 --out-dir out_batch64

test -e out/files.csv && rm  -v out/files.csv
ipython train_model/train_ssl.py --  --src-dir /Data/TCGA_LUSC/tiles/ --checkpoint --epochs 3 --moco-k 65536 --batch-size 64  --out-dir out_batch64


test -e out/files.csv && rm -v out/files.csv
ipython train_model/train_ssl.py --  --src-dir /Data/TCGA_LUSC/tiles/ --no-checkpoint --epochs 3 --moco-k 65536 --batch-size 128 --out-dir out_batch128

test -e out/files.csv && rm  -v out/files.csv
ipython train_model/train_ssl.py --  --src-dir /Data/TCGA_LUSC/tiles/ --checkpoint --epochs 3 --moco-k 65536 --batch-size 128  --out-dir out_batch128

