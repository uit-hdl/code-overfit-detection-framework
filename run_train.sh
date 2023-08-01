#!/usr/bin/env bash

set -xe

while getopts e:w:n:m:s: flag
do
    case "${flag}" in
        e) epoch=${OPTARG};;
        n) n=${OPTARG};;
        m) m=${OPTARG};;
        w) w=${OPTARG};;
        s) s=${OPTARG};;
        *) true;;
    esac
done

ipython ./train_ssl.py -- \
	--data_dir "${s:-/Data/winter_school/data_dir/}" --split_dir my_split_dir/ \
	--batch_slide_num "${n}" --cos \
	--out_dir my_output/ --epochs "${epoch}" --batch-size "${m}" \
       	--moco-t 0.07 --moco-m 0.999 \
       	--workers "${w:-16}"


