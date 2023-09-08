#!/usr/bin/env bash

while getopts e:w:n:m:o:s:c: flag
do
    case "${flag}" in
        e) epoch=${OPTARG};;
        n) n=${OPTARG};;
        m) m=${OPTARG};;
        w) w=${OPTARG};;
        s) s=${OPTARG};;
        c) c=${OPTARG};;
        o) o=${OPTARG};;
        *) true;;
    esac
done

set -xe
python ./train_model/train_ssl.py \
	--data-dir "${s:-/data/lung_scc/}" \
	--batch_slide_num "${n:-4}" --cos \
	--out-dir "${o:-/output}" --epochs "${epoch:-100}" --batch-size "${m:-64}" \
       	--moco-t 0.07 --moco-m 0.999 \
        --condition "${c:-True}" \
      	--workers "${w:-16}"
