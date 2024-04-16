#!/usr/bin/env bash
set -xe

#conda activate conditional_ssl_hist

# test -d umap_out && rm -rv umap_out 
# mkdir umap_out

for model in "/Data/winter_school/simple_embedding_biased.pkl"
# for model in "/Data/winter_school/model_default_params_1batch_3slide_only3photo.pth.tar" "/Data/winter_school/simple_embedding_biased.pkl" "/Data/winter_school/uncondition.pth.tar" "/Data/winter_school/moco_bsize200_defaults.pth.tar"
do
  name="${model##*/}"
  name="${name%%.*}"
  test -d "umap_out/${name}" && rm -rv "umap_out/${name}"
  mkdir "umap_out/${name}"
  # ipython ./feature_extraction/extract_embeddings.py -- --feature_extractor_dir "${model}" --root_dir /Data/winter_school/data_dir/ --split_dir my_split_dir/ --out_dir my_embeddings_output --subtype_model_dir my_pretrained_weights/pretrained_inception.pth.tar
  ipython ./feature_extraction/get_clusters.py -- --data_dir my_embeddings_output --cluster_type gmm --n_cluster 50 --out_dir my_cluster_output

  cp -ri my_cluster_output/* "umap_out/${name}/"
done
