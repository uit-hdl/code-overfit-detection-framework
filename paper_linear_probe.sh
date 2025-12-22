#!/usr/bin/env bash


if [[ ! -e out/balanced_dataset_top5_test.csv ]]
then
	python create_balanced_dataset_top5.py -i balanced_dataset_top5.csv -n 0.8 --out-file out/balanced_dataset_top5_train.csv --write-remainder
	mv out/balanced_dataset_top5_train.remainder.csv out/balanced_dataset_top5_test.csv
fi

tiles_root_dir=~/tiles/
for labelfile in out/balanced_dataset_top5_train.csv out/balanced_dataset_top5_test.csv
do
    suffix="${labelfile##*top5_}"
    suffix="${suffix%%.csv}"
    subdir="$PWD/out/tiles_lp_$suffix/"
    if [[ -d $subdir ]]
    then
	    echo "Tile directory already found, not moving files from CSV files"
	    continue
    fi
    mkdir $subdir
    echo "Moving tiles using $labelfile into $subdir..."
    sed 1d $labelfile | while read -r line ; do x="$(echo $line | cut -d, -f1)"; bn="$(basename $x)"; dn=$(dirname $x); dn=$(basename $dn); test -d $subdir/$dn || mkdir $subdir/$dn; cp "${tiles_root_dir}/$dn/$bn" "$subdir/$dn/$bn"; done
    header="$(head -n1 $labelfile)"
    sed -i "1s/^/$header/ ; s|^.*\(/TCGA\)|$subdir\1|" $labelfile | head
    echo "... done"
done

#ipython feature_extraction/extract_features_inceptionv4.py -- --src-dir ~/tiles_top5/ --out-dir out_top5_embeddings/ --model-pth out/models/moco/tiles/model/checkpoint_moco_tiles_0200_False_m128_n0_o0_K65536.pth.tar
