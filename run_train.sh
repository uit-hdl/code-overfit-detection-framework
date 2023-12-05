#!/usr/bin/env bash

#conda activate conditional_ssl_hist

if [[ -z "${1}" ]]
then
  SRC_DIR="${HOME}/TCGA_LUSC/preprocessed/TCGA/tiles/"
else
  SRC_DIR="${1}"
fi

SHA="$(git rev-parse HEAD)"
model_out="model_out$SHA/"
mkdir "${model_out}"

for model in m256n0o4k256True m256n0o0k256False m256n0o0k65536False
do
  m="$(basename "${model}" | command grep -Eo 'm[0-9]+' | cut -c2-)"
  n="$(basename "${model}" | command grep -Eo "n[0-9]+" | cut -c2-)"
  o="$(basename "${model}" | command grep -Eo "o[0-9]+" | cut -c2-)"
  k="$(basename "${model}" | command grep -Eo "k[0-9]+" | cut -c2-)"
  c="$(basename "${model}" | command grep -Eo "True|False")"
  echo "m:$m n:$n o:$o k:$k c:$c"
  if [[ "$c" == True ]]
  then
	  cond="--condition"
  else
	  cond="--no-condition"
  fi
  set -xe
  ipython ./train_model/train_ssl.py -- \
	--data-dir "${SRC_DIR}" \
	--batch_slide_num "${n}" \
	--batch_inst_num "${o}" \
	--out-dir "${model_out}" \
	--batch-size "${m}" \
	--moco-k "${k}" \
	--epochs 1 \
  	--moco-k "${m}" \
	"${cond}" \
	--workers 6 \
	    | tee "${model_out}/train_ssl_${m}_${n}_${o}_${c}_${k}.log"
  set +xe
done
