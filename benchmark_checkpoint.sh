#!/usr/bin/env bash

# tensorboard logs whether or not checkpoint is enabled...

base_cmd="ipython train_model/train_ssl.py -- --checkpoint --no-condition --src-dir ~/tiles_full/ --epochs 3 --moco-k 65536 --out-dir out_seq_ckpt_24workers --workers 24"
set -xe
for bs in 32 64 128 256
do
  eval "$base_cmd --batch-size $bs"
done

base_cmd="ipython train_model/train_ssl.py -- --no-checkpoint --no-condition --src-dir ~/tiles_full/ --epochs 3 --moco-k 65536 --out-dir out_seq_ckpt_24workers --workers 24"
for bs in 32 64
do
  eval "$base_cmd --batch-size $bs"
done

