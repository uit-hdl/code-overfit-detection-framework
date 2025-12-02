#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

set -e

for run in 1
do
  for sz in twenty quart half
  do
    cd ~/code-overfit-detection-framework
    labelfile="$PWD/balanced_dataset_top5_${sz}_$run.csv"
    subdir=~/tiles_${sz}_${run}
    num=2
    if [[ ${sz} -eq "quart" ]]
    then
	 num=4
    fi
    if [[ ${sz} -eq "eigth" ]]
    then
	 num=8
    fi
    if [[ ${sz} -eq "twenty" ]]
    then
	 num=20
    fi
    conda activate overfit-detection
    if [[ ! -d $subdir ]]
    then
            mkdir $subdir
            python create_balanced_dataset_top5.py -i balanced_dataset_top5.csv -n "${num}" --out-file $labelfile
	    echo "cp ..."
            cat $labelfile | sed 1d | while read -r line ; do x="$(echo $line | cut -d, -f1)"; bn="$(basename $x)"; dn=$(dirname $x); dn=$(basename $dn); test -d $subdir/$dn || mkdir $subdir/$dn; cp $x $subdir/$dn/$bn; done
            header="$(head -n1 balanced_dataset_top5.csv)"
            sed -i "1s/^/$header/ ; s#tiles/#tiles_${sz}_${run}/#" $labelfile
	    echo "done ..."
    fi

    embed_dir=$PWD/out_${sz}_${run}
    if [[ ! -d $embed_dir ]]
    then
      ipython feature_extraction/extract_features_inceptionv4.py -- --src-dir $subdir --out-dir $embed_dir --model-pth out/models/moco/tiles/model/checkpoint_moco_tiles_0200_False_m128_n0_o0_K65536.pth.tar 
      ipython feature_extraction/extract_features_phikon2.py -- --src-dir $subdir --out-dir $embed_dir
    fi
  
    conda activate feature-inspect-deep2
    cd ~/feature-inspect
    ipython ~/feature-inspect/examples/use_case_linear_probe.py -- --embeddings-path $embed_dir/inception_tiles_${sz}_${run}_embedding.zarr/ --label-file $labelfile --label-key institution --out-dir lp_final/out_${sz}_inception_$run --epochs 60 --batch-size 256
    ipython ~/feature-inspect/examples/use_case_linear_probe.py -- --embeddings-path $embed_dir/phikon_tiles_${sz}_${run}_embedding.zarr --label-file $labelfile --label-key institution --out-dir lp_final/out_${sz}_phikon_$run --epochs 60 --batch-size 256

  done
done
