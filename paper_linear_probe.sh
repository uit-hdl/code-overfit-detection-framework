#!/usr/bin/env bash

set -e

OUT_DIR=${1:-out_lp}

if [[ ! -d "${OUT_DIR}" ]]
then
  mkdir "${OUT_DIR}"
fi

if [[ ! -e ${OUT_DIR}/balanced_dataset_top5_test.csv ]]
then
  set -x
    python create_balanced_dataset_top5.py -i balanced_dataset_top5.csv -n 0.85 --out-file ${OUT_DIR}/balanced_dataset_top5_train.csv --write-remainder
    mv ${OUT_DIR}/balanced_dataset_top5_train.remainder.csv ${OUT_DIR}/balanced_dataset_top5_test.csv
    set +x
fi

tiles_root_dir=~/tiles/
for labelfile in ${OUT_DIR}/balanced_dataset_top5_train.csv ${OUT_DIR}/balanced_dataset_top5_test.csv
do
    suffix="${labelfile##*top5_}"
    suffix="${suffix%%.csv}"
    subdir="$PWD/${OUT_DIR}/tiles_lp_$suffix/"
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

if [[ ! -e ${OUT_DIR}/top5_embeddings_train/inception_tiles_lp_train_embedding.zarr ]]
then
  set -x
  ipython feature_extraction/extract_features_inceptionv4.py -- --src-dir ${OUT_DIR}/tiles_lp_train --out-dir ${OUT_DIR}/top5_embeddings_train/ --model-pth out/models/moco/tiles/model/checkpoint_moco_tiles_0200_False_m128_n0_o0_K65536.pth.tar
  ipython feature_extraction/extract_features_phikon2.py -- --src-dir ${OUT_DIR}/tiles_lp_train --out-dir ${OUT_DIR}/top5_embeddings_train/
  set +x
fi

if [[ ! -e ${OUT_DIR}/top5_embeddings_test/inception_tiles_lp_test_embedding.zarr ]]
then
  set -x
  ipython feature_extraction/extract_features_inceptionv4.py -- --src-dir ${OUT_DIR}/tiles_lp_test --out-dir ${OUT_DIR}/top5_embeddings_test/ --model-pth out/models/moco/tiles/model/checkpoint_moco_tiles_0200_False_m128_n0_o0_K65536.pth.tar
  ipython feature_extraction/extract_features_phikon2.py -- --src-dir ${OUT_DIR}/tiles_lp_test --out-dir ${OUT_DIR}/top5_embeddings_test/
  set +x
fi

for modelname in inception phikon
do
  if [[ ! -e "${OUT_DIR}/prediction_${modelname}_0.csv" ]]
  then
    set -x
    python linear_probe.py --embeddings-path "${OUT_DIR}/top5_embeddings_train/${modelname}_tiles_lp_train_embedding.zarr/" --test-embeddings-path "${OUT_DIR}/top5_embeddings_test/${modelname}_tiles_lp_test_embedding.zarr/" --label-key institution --tensorboard-name "lp_${modelname}_100" --out-dir "${OUT_DIR}" --epochs 20
    set +x
    cd ${OUT_DIR}
    for file in prediction_model*
    do
      x="$(echo $file | sed s/model/$modelname/g)"
      mv $file "${x}"
    done
    mv accuracies.csv "${modelname}_accuracies.csv"
    cd -
  fi
done

if [[ ! -e  "${OUT_DIR}/t_tests_100.txt" ]]
then
  set -x
  python t_test_accuracies.py ${OUT_DIR}/inception_accuracies.csv ${OUT_DIR}/phikon_accuracies.csv | tee "${OUT_DIR}/t_tests_100.txt"
  set +x
fi


for modelname in phikon inception
do
  for sz in 0.05 0.25
  do
    if [[ ! -e ${OUT_DIR}/accuracies_${modelname}_${sz}.csv ]]
    then
    set -x
    python linear_probe_train_montecarlo.py \
      --embeddings-path "out_lp/top5_embeddings_train/${modelname}_tiles_lp_train_embedding.zarr/" \
      --test-embeddings-path "out_lp/top5_embeddings_test/${modelname}_tiles_lp_test_embedding.zarr/" \
      --label-key institution \
      --tensorboard-name lp_montecarlo_${modelname}_${sz} \
      --out-dir ${OUT_DIR} \
      --epochs 60 \
      --subset-size ${sz} \
      --rounds 100
    set +x
    mv "${OUT_DIR}/accuracies_${sz}.csv" "${OUT_DIR}/accuracies_${modelname}_${sz}.csv"
    fi
  done
done


echo "done."
