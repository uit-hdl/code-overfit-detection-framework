#!/usr/bin/env bash

#conda activate conditional_ssl_hist

if [[ -z "${1}" ]]
then
  SRC_DIR="${HOME}/TCGA_LUSC/preprocessed/TCGA/tiles/"
else
  SRC_DIR="${1}"
fi

mkdir analysis_out/

for model in out/MoCo/tiles/model/*.tar
do
  m="$(basename "${model}" | command grep -Eo 'm[0-9]+')"
  n="$(basename "${model}" | command grep -Eo "n[0-9]+")"
  o="$(basename "${model}" | command grep -Eo "o[0-9]+")"
  c="$(basename "${model}" | command grep -Eo "True|False")"
  echo "$m $n $c $o"
  out_dir="analysis_out/out${m}${n}${o}${c}"
  set -xe
  ipython feature_extraction/extract_embeddings.py -- --src_dir "${SRC_DIR}" --feature_extractor "${model}" --out_dir "${out_dir}"
  ipython feature_extraction/get_clusters.py -- --embeddings_path "${out_dir}/MoCo/tiles/embeddings/test_tiles_embedding.pkl" --out_dir "${out_dir}"
  ipython survival_models/cox.py --  --embeddings_dir "${out_dir}/MoCo/tiles/embeddings/" --out_dir "${out_dir}/survival"
  set +xe
done
