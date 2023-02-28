#!/usr/bin/env -S python -m IPython

from itertools import accumulate, combinations, product
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff, pdist, cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import argparse
import frechetdist
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import operator
import pandas as pd
import pickle
import sys
import umap
import networkx as nx


parser = argparse.ArgumentParser(description='Get cluster features')
reducer = umap.UMAP(random_state=42)

parser.add_argument('--data_dir', default='./', type=str)
parser.add_argument('--cluster_type', default='gmm', type=str)
parser.add_argument('--n_cluster', default=50, type=int)
parser.add_argument('--out_dir', default='./', type=str)


args = parser.parse_args()

# train_features = pickle.load(open(args.data_dir + '/test_embedding.pkl', 'rb'))
# train_features_flattened = np.concatenate(list(train_features.values()), axis=0)
# cluster = GaussianMixture(n_components=args.n_cluster).fit(train_features_flattened)
# pickle.dump(cluster, open(args.out_dir + '/gmm_{}.pkl'.format(args.n_cluster), 'wb'))

def umap_slice(slide_ids):
    train_features = pickle.load(open(args.data_dir + '/test_embedding.pkl', 'rb'))
    keys = []
    for i,x in enumerate(sorted(train_features.keys())):
        if i in slide_ids:
            continue
        else:
            keys.append(x)
    for x in keys:
        del train_features[x]
    train_features_flattened = np.concatenate(list(train_features.values()), axis=0)
    umap_projection = reducer.fit_transform(train_features_flattened)
    slices = list(accumulate([0] + [len(y) for y in train_features.values()], operator.add))
    fig, ax = plt.subplots()
    slide_sets = [[]] * len(train_features.keys())
# [i]nterval_[s]tart, [e]nd
    for slide_number,(i_s,i_e) in enumerate(zip(slices, slices[1:])):
        ax.scatter(umap_projection[i_s:i_e, 0], umap_projection[i_s:i_e, 1], label = f"Slide {slide_number+1}", alpha=.5)
        slide_sets[slide_number] = umap_projection[i_s:i_e]
    for i,ss in enumerate(slide_sets):
        savetxt(os.path.join(args.out_dir, 'slide%d.csv' % (i+1)), ss, delimiter=',')
    ax.legend()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # only one of the below, not both
    plt.show()
    # plt.savefig(os.path.join('.', 'umap_output.png'))
    hausdorf_l = np.zeros((len(train_features.keys()), len(train_features.keys())))
    np.fill_diagonal(hausdorf_l, np.nan)
    frechet_l = np.zeros((len(train_features.keys()), len(train_features.keys())))
    np.fill_diagonal(frechet_l, np.nan)
# from Data Representativity for Machine Learning and AI Systems
    wasserstein_l = np.zeros((len(train_features.keys()), len(train_features.keys())))
    np.fill_diagonal(wasserstein_l, np.nan)
# TODO: shannon coverage? look at all other values and measure coverage of those
    for ((i,left),(j, right)) in combinations(enumerate(slide_sets), 2):
        haus_d = max(directed_hausdorff(left, right)[0], directed_hausdorff(right, left)[0])
        hausdorf_l[i][j] = haus_d
        hausdorf_l[j][i] = haus_d


# TODO: frechet
        l_s = left.shape[0]
        r_s = right.shape[0]

        # Extend the smallest of left or right with the contents from the bigger one (giving distance=0)
        # FIXME: this is equivalent to dropping of points................
        if l_s < r_s:
            # tail = right[-(r_s-l_s):]
            # left = np.append(left, tail, axis=0)
            right = right[-l_s:]
        else:
            # tail = left[-(l_s-r_s):]
            # right = np.append(right, tail, axis=0)
            left = left[-r_s:]

        frdist = frechetdist.frdist(left, right)
        frechet_l[i][j] = frdist
        frechet_l[j][i] = frdist

        #https://stackoverflow.com/a/57563383
        cd = cdist(left, right)
        assignment = linear_sum_assignment(cd)
        wd = cd[assignment].sum() / left.shape[0]
        wasserstein_l[i][j] = wd
        wasserstein_l[j][i] = wd


    labels = list(map(lambda l: 'Slide %d' % (l+1), range(hausdorf_l.shape[0]))) + ['mean']
    mean_hausdorf = np.nanmean(hausdorf_l)
    hausdorf_l = np.vstack([hausdorf_l, np.nanmean(hausdorf_l, axis=1)])
    hausdorf_l = np.pad(hausdorf_l, ((0,0),(0,1)), mode='constant', constant_values=mean_hausdorf)
    hdf = pd.DataFrame(hausdorf_l, columns=labels).to_csv(os.path.join(args.out_dir, 'hausdorf.csv'))
    print(hdf)
    mean_frechet = np.nanmean(frechet_l)
    frechet_l = np.vstack([frechet_l, np.nanmean(frechet_l, axis=1)])
    frechet_l = np.pad(frechet_l, ((0,0),(0,1)), mode='constant', constant_values=mean_frechet)
    hdf = pd.DataFrame(frechet_l, columns=labels).to_csv(os.path.join(args.out_dir, 'frechet.csv'))
    print(hdf)
    mean_wd = np.nanmean(wasserstein_l)
    wasserstein_l = np.vstack([wasserstein_l, np.nanmean(wasserstein_l, axis=1)])
    wasserstein_l = np.pad(wasserstein_l, ((0,0),(0,1)), mode='constant', constant_values=mean_wd)
    hdf = pd.DataFrame(wasserstein_l, columns=labels).to_csv(os.path.join(args.out_dir, 'wasserstein.csv'))
    print(hdf)

    metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
# metrics_l = np.zeros((len(train_features.keys())), len(metrics))
    metrics_l = np.zeros((len(train_features.keys()), len(metrics)))
    for ((i,points), (j,metric)) in product(enumerate(slide_sets), enumerate(metrics)):
        metrics_l[i][j] = np.mean(pdist(points, metric))

    df = pd.DataFrame(metrics_l, columns=metrics)
    df.to_csv(os.path.join(args.out_dir, 'out.csv'), sep='\t', encoding='utf-8')
    print(df)
# plt.show()

train_features = pickle.load(open(args.data_dir + '/test_embedding.pkl', 'rb'))
len_keys = len(train_features.keys())
# for s in range(0, len_keys-8, 8):
for s in combinations(range(len_keys), 8):
    umap_slice(list(s))
    break
# umap_slice(0, 8)


