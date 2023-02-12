import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pickle
import numpy as np
import sys
import umap
import matplotlib.pyplot as plt
from itertools import accumulate
import operator


parser = argparse.ArgumentParser(description='Get cluster features')
reducer = umap.UMAP(random_state=42)

parser.add_argument('--data_dir', default='./', type=str)
parser.add_argument('--cluster_type', default='gmm', type=str)
parser.add_argument('--n_cluster', default=50, type=int)
parser.add_argument('--out_dir', default='./', type=str)


args = parser.parse_args()

train_features = pickle.load(open(args.data_dir + '/train_embedding.pkl', 'rb'))
train_features_flattened = np.concatenate(list(train_features.values()), axis=0)
if args.cluster_type == "kmeans":
    print('kmeans')
    cluster = KMeans(n_clusters=n_cluster).fit(train_features)
    pickle.dump(cluster, open(args.out_dir + '/kmeans_{}.pkl'.format(args.n_cluster), 'wb'))
else:
    print('gmm')
    # cluster = GaussianMixture(n_components=args.n_cluster).fit(train_features_flattened)
    umap_projection = reducer.fit_transform(train_features_flattened)
    slices = list(accumulate([0] + [len(y) for y in train_features.values()], operator.add))
    fig, ax = plt.subplots()
    for slide_number,(i_s,i_e) in enumerate(zip(slices, slices[1:])):
        ax.scatter(umap_projection[i_s:i_e, 0], umap_projection[i_s:i_e, 1], label = f"Slide {slide_number}", alpha=.5)
    ax.legend()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('UMAP projection of tile representations', fontsize=20)
    plt.savefig(os.path.join('doc', 'comparison_slides_umap.png'))
    # plt.show()
    # pickle.dump(cluster, open(args.out_dir + '/gmm_{}.pkl'.format(args.n_cluster), 'wb'))


