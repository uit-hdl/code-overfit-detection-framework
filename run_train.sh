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

#for model in out/MoCo/tiles/model/*.tar
#for model in out/MoCo/tiles/model/checkpoint_MoCo_tiles_0200_True_m128_n4.pth.tar out/MoCo/tiles/model/checkpoint_MoCo_tiles_0200_False_m128_n0_o0.pth.tar
for model in m256n0o4True m256n0o0False # m128n0o4True m128n0o0False 
do
  m="$(basename "${model}" | command grep -Eo 'm[0-9]+' | cut -c2-)"
  n="$(basename "${model}" | command grep -Eo "n[0-9]+" | cut -c2-)"
  o="$(basename "${model}" | command grep -Eo "o[0-9]+" | cut -c2-)"
  c="$(basename "${model}" | command grep -Eo "True|False")"
  echo "m:$m n:$n o:$o c:$c"
  set -xe
  ipython ./train_model/train_ssl.py -- \
	--data-dir "${SRC_DIR}" \
	--batch_slide_num "${n}" \
	--batch_inst_num "${o}" \
	--out-dir "${model_out}" 
        --batch-size "${m}" \
        --condition "${c}" \
      	--workers 6
  set +xe
done

set -xe
