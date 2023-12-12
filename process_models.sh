#!/usr/bin/env bash

#conda activate conditional_ssl_hist

if [[ -z "${1}" ]]
then
  SRC_DIR="${HOME}/TCGA_LUSC/preprocessed/TCGA/tiles/"
else
  SRC_DIR="${1}"
fi

if [[ -z "${2}" ]]
then
  model_out_dir="out"
else
  model_out_dir="${2}"
fi

SHA="$(git rev-parse HEAD)"
analysis_dir="analysis_out_$SHA/"
mkdir "${analysis_dir}"

ipython plot_model_learning.py -- "${model_out_dir}"/MoCo/tiles/data/
#for model in "${model_out_dir}"/MoCo/tiles/model/*0200*.tar
for model in "${model_out_dir}"/MoCo/tiles/model/*0180*False*.tar
do
  m="$(basename "${model}" | command grep -Eo 'm[0-9]+')"
  n="$(basename "${model}" | command grep -Eo "n[0-9]+")"
  o="$(basename "${model}" | command grep -Eo "o[0-9]+")"
  K="$(basename "${model}" | command grep -Eo "K[0-9]+")"
  c="$(basename "${model}" | command grep -Eo "True|False")"
  echo "$m $n $c $K $o"
  out_dir="${analysis_dir}/out${m}${n}${K}${o}${c}"
  set -xe
  ipython feature_extraction/extract_embeddings.py -- --src_dir "${SRC_DIR}" --feature_extractor "${model}" --out_dir "${out_dir}"
  ipython feature_extraction/get_clusters.py -- --embeddings_path "${out_dir}/MoCo/tiles/embeddings/test_tiles_embedding.pkl" --out_dir "${out_dir}" --number_of_images 20 --histogram_bins 20
  ipython feature_extraction/get_clusters.py -- --embeddings_path "${out_dir}/MoCo/tiles/embeddings/test_tiles_embedding.pkl" --out_dir "${out_dir}" --number_of_images 2999 --histogram_bins 2999
  #ipython survival_models/cox.py --  --embeddings_dir "${out_dir}/MoCo/tiles/embeddings/" --out_dir "${out_dir}/survival"
  set +xe
done
