#!/usr/bin/env bash

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

set -xe
ipython ./train_ssl.py -- \
	--data_dir "${s:-/Data/winter_school/data_dir/}" \
	--batch_slide_num "${n}" --cos \
	--out_dir ../out/ --epochs "${epoch}" --batch-size "${m}" \
       	--moco-t 0.07 --moco-m 0.999 \
       	--workers "${w:-16}"


