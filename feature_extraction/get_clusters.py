import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pickle
import numpy as np
import sys
import umap
import matplotlib.pyplot as plt
from itertools import accumulate, combinations, product
import operator
from scipy.spatial.distance import directed_hausdorff, pdist
import pandas as pd


parser = argparse.ArgumentParser(description='Get cluster features')
reducer = umap.UMAP(random_state=42)

parser.add_argument('--data_dir', default='./', type=str)
parser.add_argument('--cluster_type', default='gmm', type=str)
parser.add_argument('--n_cluster', default=50, type=int)
parser.add_argument('--out_dir', default='./', type=str)


args = parser.parse_args()

train_features = pickle.load(open(args.data_dir + '/train_embedding.pkl', 'rb'))
train_features_flattened = np.concatenate(list(train_features.values()), axis=0)
# if args.cluster_type == "kmeans":
#     print('kmeans')
#     cluster = KMeans(n_clusters=n_cluster).fit(train_features)
#     pickle.dump(cluster, open(args.out_dir + '/kmeans_{}.pkl'.format(args.n_cluster), 'wb'))
# else:
print('gmm')
# cluster = GaussianMixture(n_components=args.n_cluster).fit(train_features_flattened)
umap_projection = reducer.fit_transform(train_features_flattened)
slices = list(accumulate([0] + [len(y) for y in train_features.values()], operator.add))
fig, ax = plt.subplots()
slide_sets = [[]] * len(train_features.keys())
# [i]nterval_[s]tart, [e]nd
for slide_number,(i_s,i_e) in enumerate(zip(slices, slices[1:])):
    ax.scatter(umap_projection[i_s:i_e, 0], umap_projection[i_s:i_e, 1], label = f"Slide {slide_number}", alpha=.5)
    slide_sets[slide_number] = umap_projection[i_s:i_e]
ax.legend()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('UMAP projection of tile representations', fontsize=20)
plt.savefig(os.path.join('doc', 'comparison_slides_umap.png'))

hausdorf_l = np.zeros((len(train_features.keys()), len(train_features.keys())))
for ((i,left),(j, right)) in combinations(enumerate(slide_sets), 2):
    d = max(directed_hausdorff(left, right)[0], directed_hausdorff(right, left)[0])
    hausdorf_l[i][j] = d
    hausdorf_l[j][i] = d
mean_hausdorf = np.mean(hausdorf_l)
print(hausdorf_l)
print(mean_hausdorf)

metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
# metrics_l = np.zeros((len(train_features.keys())), len(metrics))
metrics_l = np.zeros((len(train_features.keys()), len(metrics)))
for ((i,points), (j,metric)) in product(enumerate(slide_sets), enumerate(metrics)):
    metrics_l[i][j] = np.mean(pdist(points, metric))
df = pd.DataFrame(metrics_l, columns=metrics)
df.to_csv('out.csv', sep='\t', encoding='utf-8')
# plt.show()
# pickle.dump(cluster, open(args.out_dir + '/gmm_{}.pkl'.format(args.n_cluster), 'wb'))


