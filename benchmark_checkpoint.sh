#!/usr/bin/env bash

base_cmd="ipython train_model/train_ssl.py -- --no-condition --src-dir ~/tiles/ --epochs 10 --moco-k 65536 --out-dir out_seq_ckpt"
for bs in 32 64 128 256 512
do
  echo $bs
done