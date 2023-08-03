#!/usr/bin/env -S python -m IPython

import argparse
import operator
import os
import pickle
from itertools import accumulate

import matplotlib.pyplot as plt
import numpy as np
import umap
from numpy import savetxt


parser = argparse.ArgumentParser(description='Get cluster features')
reducer = umap.UMAP(random_state=42)

parser.add_argument('--embeddings_dir', default='./', type=str, help="location of embeddings")
parser.add_argument('--cluster_type', default='gmm', type=str)
parser.add_argument('--n_cluster', default=50, type=int)
parser.add_argument('--out_dir', default='./out', type=str)

args = parser.parse_args()

#cluster = GaussianMixture(n_components=args.n_cluster).fit(train_features_flattened)
# pickle.dump(cluster, open(args.out_dir + '/gmm_{}.pkl'.format(args.n_cluster), 'wb'))

def umap_slice(names, features):
    values = [features[name] for name in names]
    features_flattened = np.concatenate(values, axis=0)
    umap_projection = reducer.fit_transform(features_flattened)
    slices = list(accumulate([0] + [len(y) for y in values], operator.add))
    fig, ax = plt.subplots()
    slide_sets = [[]] * len(names())
    keys = list(features.keys())
# [i]nterval_[s]tart, [e]nd
    for slide_number,(i_s,i_e) in enumerate(zip(slices, slices[1:])):
        ax.scatter(umap_projection[i_s:i_e, 0], umap_projection[i_s:i_e, 1], label = f"Slide {keys[slide_number]}", alpha=.5)
        slide_sets[slide_number] = umap_projection[i_s:i_e]
    for i, ss in enumerate(slide_sets):
        savetxt(os.path.join(args.out_dir, '{}.csv'.format(keys[i])), ss, delimiter=',')
    ax.legend()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # only one of the below, not both
    plt.show()
    #plt.savefig(os.path.join('.', 'umap_output.png'))

train_features = pickle.load(open(args.embeddings_dir + '/test_embedding.pkl', 'rb'))
train_features_flattened = np.concatenate(list(train_features.values()), axis=0)
train_features = pickle.load(open(args.embeddings_dir + '/test_embedding.pkl', 'rb'))
keys_sorted = list(sorted(train_features.keys()))
print ("There are {} images in the dataset".format(len(keys_sorted)))
# for s in combinations(keys_sorted, 3):
#     train_features = pickle.load(open(args.data_dir + '/test_embedding.pkl', 'rb'))
#     y = {k : train_features[k] for k in s}
#     # get a subset of train_features from the keys in s
#     subset_features = {k : train_features[k] for k in y}
#     umap_slice(subset_features)
umap_slice(['TCGA-43-8115-01A-01-BS1', 'TCGA-34-8456-01A-01-BS1', 'TCGA-68-A59J-01A-02-TSB'], train_features)
# umap_slice(train_features)
