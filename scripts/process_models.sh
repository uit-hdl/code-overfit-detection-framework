#!/usr/bin/env bash

#conda activate conditional_ssl_hist

if [[ -z "${1}" ]]
then
  SRC_DIR="${HOME}/TCGA_LUSC/tiles/"
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

if [[ -z "${4}" ]]
then
  SRC_ALT_DIR="${SRC_DIR}"
else
  SRC_ALT_DIR="${4}"
fi

bsrc="$(basename $SRC_DIR)"
balt="$(basename $SRC_ALT_DIR)"


SHA="$(git rev-parse HEAD)"
analysis_dir="analysis_out_$SHA/"
test -d "${analysis_dir}" || mkdir -p "${analysis_dir}"

#for model in "${model_out_dir}"/MoCo/tiles/model/*0200*.tar
#for model in "${model_out_dir}"/MoCo/tiles/model/*0180*False*.tar
#for model in "${model_out_dir}"/MoCo/*/model/*0040*False*.tar
for model in $(find out* -name \*checkpoint\*.pth.tar -type f)
do
  m="$(basename "${model}" | command grep -Eo 'm[0-9]+')"
  n="$(basename "${model}" | command grep -Eo "n[0-9]+")"
  o="$(basename "${model}" | command grep -Eo "o[0-9]+")"
  K="$(basename "${model}" | command grep -Eo "K[0-9]+")"
  c="$(basename "${model}" | command grep -Eo "True|False")"
  e="$(basename "${model}" | command grep -Eo "0[0-9]+")"
  d="$(basename "${model}" | command grep -Eo "tiles[0-9A-Za-z-]?")"
  out_dir="${analysis_dir}/out_${d}e${e}${m}${n}${K}${o}${c}"
  echo "e$e $d $m $n $c $K $o $out_dir $model"
  set -xe
  # TODO: produce no-color-normalizatio CPTAC
  test -e "${out_dir}"/inceptionv4/$(basename ${model})/cptac_*_embedding.pkl || \
	  ipython feature_extraction/extract_embeddings_cptac.py -- --src-dir "${CPTAC_SRC_DIR}" --feature-extractor "${model}" --out-dir "${out_dir}"
  test -e "${out_dir}"/inceptionv4/$(basename ${model})/test_${bsrc}_embedding.pkl || \
	  ipython feature_extraction/extract_embeddings.py -- --src-dir "${SRC_DIR}" --feature-extractor "${model}" --out-dir "${out_dir}"

  test -d "${out_dir}"/web_${bsrc}/ || \
	  ipython feature_extraction/get_clusters.py -- --embeddings-path-val "${out_dir}"/inceptionv4/$(basename ${model})/val_${bsrc}_embedding.pkl --embeddings-path-train "${out_dir}"/inceptionv4/$(basename ${model})/train_${bsrc}_embedding.pkl --embeddings-path-test "${out_dir}"/inceptionv4/$(basename ${model})/test_${bsrc}_embedding.pkl --out-dir "${out_dir}" --number-of-images 2999 --histogram-bins 20

  #test -d "${out_dir}"/MoCo/${bsrc}/model/relabelled_my_inst*${bsrc}*/ || \
	  #ipython train_model/relabelling.py -- --epochs 5 --src-dir "${SRC_DIR}" --out-dir "${out_dir}" --feature_extractor "${model}" --label-key "my_inst"
  #relabelled_model="$(find "${out_dir}"/MoCo/*/model/relabelled_my_inst*${bsrc}*/ -name \*network_epoch\=5.pt -type f)"
  #test -e "${out_dir}"/relabelled_my_inst_*${bsrc}*Institution_*fairness.csv || \
	  #ipython ./misc/fairness_relabelled_my_inst.py -- --feature_extractor "${relabelled_model}" --src_dir "${SRC_DIR}" --out_dir "${out_dir}"

  test -d "${out_dir}"/MoCo/*/model/relabelled_Sample*${bsrc}*/ || \
	  ipython train_model/relabelling.py -- --epochs 10 --src-dir "${SRC_DIR}" --out-dir "${out_dir}" --feature_extractor "${model}" --label-key "Sample Type"
  relabelled_model="$(find "${out_dir}"/MoCo/*/model/relabelled_Sample*${bsrc}*/ -name \*network_epoch\=10.pt -type f)"
  test -e "${out_dir}"/relabelled_Sample*${bsrc}*Institution_fairness.csv || \
	  ipython ./misc/fairness.py -- --feature_extractor "${relabelled_model}" --src_dir "${SRC_DIR}" --out_dir "${out_dir}"
  # TODO: produce no-color-normalizatio CPTAC
  test -d "${out_dir}"/validation_tiles || \
	  ipython ./misc/external_validation.py -- --src-dir "${CPTAC_SRC_DIR}" --out-dir "${out_dir}" --feature_extractor "${relabelled_model}" \

  # cover all ground
  if [[ ! "${SRC_ALT_DIR}" == "${SRC_DIR}" ]]
  then
	  test -e "${out_dir}"/inceptionv4/$(basename ${model})/test_${balt}_embedding.pkl || \
		  ipython feature_extraction/extract_embeddings.py -- --src-dir "${SRC_ALT_DIR}" --feature-extractor "${model}" --out-dir "${out_dir}"

	  test -d "${out_dir}"/web_${balt}/ || \
		  ipython feature_extraction/get_clusters.py -- --embeddings-path-val "${out_dir}"/inceptionv4/$(basename ${model})/val_${balt}_embedding.pkl --embeddings-path-train "${out_dir}"/inceptionv4/$(basename ${model})/train_${balt}_embedding.pkl --embeddings-path-test "${out_dir}"/inceptionv4/$(basename ${model})/test_${balt}_embedding.pkl --out-dir "${out_dir}" --number-of-images 2999 --histogram-bins 20

	  #test -d "${out_dir}"/MoCo/${balt}/model/relabelled_my_inst*${balt}*/ || \
		  #ipython train_model/relabelling.py -- --epochs 5 --src-dir "${SRC_ALT_DIR}" --out-dir "${out_dir}" --feature_extractor "${model}" --label-key "my_inst"
	  relabelled_model="$(find "${out_dir}"/MoCo/*/model/relabelled_my_inst*${balt}*/ -name \*network_epoch\=5.pt -type f)"
	  #test -e "${out_dir}"/relabelled_my_inst_*${balt}*Institution_*fairness.csv || \
		  #ipython ./misc/fairness_relabelled_my_inst.py -- --feature_extractor "${relabelled_model}" --src_dir "${SRC_ALT_DIR}" --out_dir "${out_dir}"

	  test -d "${out_dir}"/MoCo/*/model/relabelled_Sample*${balt}*/ || \
		  ipython train_model/relabelling.py -- --epochs 10 --src-dir "${SRC_ALT_DIR}" --out-dir "${out_dir}" --feature_extractor "${model}" --label-key "Sample Type"
	  relabelled_model="$(find "${out_dir}"/MoCo/*/model/relabelled_Sample*${balt}*/ -name \*network_epoch\=10.pt -type f)"
	  test -e "${out_dir}"/relabelled_Sample*${balt}*Institution_fairness.csv || \
		  ipython ./misc/fairness.py -- --feature_extractor "${relabelled_model}" --src_dir "${SRC_ALT_DIR}" --out_dir "${out_dir}"
  fi


  set +xe
done
