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

if [[ -z "${3}" ]]
then
  CPTAC_SRC_DIR="${HOME}/CPTAC/tiles"
else
  CPTAC_SRC_DIR="${3}"
fi

SHA="$(git rev-parse HEAD)"
analysis_dir="analysis_out_$SHA/"
test -d "${analysis_dir}" || mkdir -p "${analysis_dir}"

#for model in "${model_out_dir}"/MoCo/tiles/model/*0200*.tar
#for model in "${model_out_dir}"/MoCo/tiles/model/*0180*False*.tar
for model in "${model_out_dir}"/MoCo/*/model/*0200*False*.tar
do
  m="$(basename "${model}" | command grep -Eo 'm[0-9]+')"
  n="$(basename "${model}" | command grep -Eo "n[0-9]+")"
  o="$(basename "${model}" | command grep -Eo "o[0-9]+")"
  K="$(basename "${model}" | command grep -Eo "K[0-9]+")"
  c="$(basename "${model}" | command grep -Eo "True|False")"
  e="$(basename "${model}" | command grep -Eo "0[0-9]+")"
  out_dir="${analysis_dir}/out_e${e}${m}${n}${K}${o}${c}"
  echo "e$e $m $n $c $K $o $out_dir $model"
  set -xe
  test -e "${out_dir}"/inceptionv4/$(basename ${model})/cptac_*_embedding.pkl || \
	  ipython feature_extraction/extract_embeddings_cptac.py -- --src-dir "${CPATC_SRC_DIR}" --feature-extractor "${model}" --out-dir "${out_dir}"
  test -e "${out_dir}"/inceptionv4/$(basename ${model})/test_*_embedding.pkl || \
	  ipython feature_extraction/extract_embeddings.py -- --src-dir "${SRC_DIR}" --feature-extractor "${model}" --out-dir "${out_dir}"

  test -d "${out_dir}"/web/ || \
	  ipython feature_extraction/get_clusters.py -- --embeddings-path-val "${out_dir}"/inceptionv4/$(basename ${model})/val_*_embedding.pkl --embeddings-path-train "${out_dir}"/inceptionv4/$(basename ${model})/train_*_embedding.pkl --embeddings-path-test "${out_dir}"/inceptionv4/$(basename ${model})/test_*_embedding.pkl --out-dir "${out_dir}" --number-of-images 2999 --histogram-bins 20

  test -d "${out_dir}"/MoCo/*/model/relabelled_my_inst*/ || \
	  ipython train_model/relabelling.py -- --epochs 5 --src-dir "${SRC_DIR}" --out-dir "${out_dir}" --feature_extractor "${model}" --label-key "my_inst"
  relabelled_model="$(find "${out_dir}"/MoCo/*/model/relabelled_my_inst*/ -name \*5.pt -type f)"
  test -e "${out_dir}"/relabelled_my_inst_*Institution_*fairness.csv || \
	  ipython ./misc/fairness_relabelled_my_inst.py -- --feature_extractor "${relabelled_model}" --src_dir "${SRC_DIR}" --out_dir "${out_dir}"

  test -d "${out_dir}"/MoCo/*/model/relabelled_Sample*/ || \
	  ipython train_model/relabelling.py -- --epochs 10 --src-dir "${SRC_DIR}" --out-dir "${out_dir}" --feature_extractor "${model}" --label-key "Sample Type"
  relabelled_model="$(find "${out_dir}"/MoCo/*/model/relabelled_Sample*/ -name \*10.pt -type f)"
  test -e "${out_dir}"/relabelled_Sample*Institution_fairness.csv || \
	  ipython ./misc/fairness.py -- --feature_extractor "${relabelled_model}" --src_dir "${SRC_DIR}" --out_dir "${out_dir}"
  test -e "${out_dir}"/validation* || \
	  ipython ./misc/external_validation.py -- --src-dir "${CPTAC_SRC_DIR}" --out-dir "${out_dir}" --feature_extractor "${relabelled_model}" \


  set +xe
done
