#!/usr/bin/env bash

set -xe

while getopts e:w:n:m: flag
do
    case "${flag}" in
        e) epoch=${OPTARG};;
        n) n=${OPTARG};;
        m) m=${OPTARG};;
        w) w=${OPTARG};;
        *) true;;
    esac
done

# torchrun ./train_ssl.py --data_dir /Data/winter_school/data_dir/ --split_dir my_split_dir/ --batch_slide_num "${n}" --cos --out_dir my_output/ --gpu 0 --epochs "${epoch}" --batch-size "${m}" --moco-t 0.07 --moco-m 0.999
ipython ./train_ssl.py -- --data_dir /Data/winter_school/data_dir/ --split_dir my_split_dir/ --batch_slide_num "${n}" --cos --out_dir my_output/ --epochs "${epoch}" --batch-size "${m}" --moco-t 0.07 --moco-m 0.999 --workers "${w:-16}"
# ipython ./train_ssl.py -- --data_dir /Data/winter_school/small_data_dir/ --split_dir my_split_dir/ --batch_slide_num "${n}" --cos --out_dir my_output/ --epochs "${epoch}" --batch-size "${m}" --moco-t 0.07 --moco-m 0.999 --workers "${w:-16}"


