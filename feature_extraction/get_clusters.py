#!/usr/bin/env -S python -m IPython

import argparse
import pandas as pd
from bokeh.plotting import show as show_interactive
from bokeh.models import ColumnDataSource, OpenURL, TapTool, WheelZoomTool
from sklearn.mixture import GaussianMixture
import operator
import os
import pickle
from itertools import accumulate

import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.plot
from numpy import savetxt
from pathlib import Path

def ensure_dir_exists(path):
    dest_dir = os.path.dirname(path)
    if not os.path.exists(dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)



parser = argparse.ArgumentParser(description='Get cluster features')
reducer = umap.UMAP(random_state=42)

parser.add_argument('--embeddings_path', default='./out/MoCo/embeddings/test_lungscc_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
parser.add_argument('--clinical_path', default='./annotations/clinical.tsv', type=str, help="location of file containing clinical data")
parser.add_argument('--cluster', default=False, type=bool, action=argparse.BooleanOptionalAction,
                    metavar='O', help='whether to cluster or not', dest='do_cluster')
parser.add_argument('--n_cluster', default=50, type=int)
parser.add_argument('--out_dir', default='./out', type=str)

args = parser.parse_args()

# color by gender

def umap_slice(names, features, cluster, clinical, out_dir):
    values = [[x[0] for x in features[name]] for name in names]
    cluster_labels = [cluster.predict(y) for y in values]
    cluster_labels = [item for sublist in cluster_labels for item in sublist]
    tile_names = [[x[1] for x in features[name]] for name in names]
    tile_names = ["file:///" + x for x in np.concatenate(tile_names, axis=0)]
    features_flattened = np.concatenate(values, axis=0)
    umap_projection = reducer.fit_transform(features_flattened)
    mapper = reducer.fit(features_flattened)
    slices = list(accumulate([0] + [len(y) for y in values], operator.add))
    names_labels = [[names[i]] * (i_e - i_s) for i,(i_s,i_e) in enumerate(zip(slices, slices[1:]))]
    names_labels = [item for sublist in names_labels for item in sublist]
    case_submitter_ids = ["-".join(name.split("-")[:3]) for name in names_labels]
    gender_labels = [clinical['gender'][case][0] for case in case_submitter_ids]

    hover_data = pd.DataFrame({'index': np.arange(len(features_flattened)),
                               'cluster_id': cluster_labels,
                               'gender': gender_labels,
                               'slide': names_labels,
                               'image_url': tile_names,
                               })
    p = umap.plot.interactive(mapper, labels=gender_labels, hover_data=hover_data, point_size=7)
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@image_url" height="150" alt="@slide" width="150"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@slide</span>
            </div>
            <div>
                <span style="font-size: 15px;">GMM Cluster:</span>
                <span style="font-size: 15px;">@cluster_id</span>
            </div>
            <div>
                <span style="font-size: 15px;">Gender:</span>
                <span style="font-size: 15px;">@gender</span>
            </div>
            <div>
                <span style="font-size: 15px;">Location</span>
                <span style="font-size: 10px; color: #696;">($x, $y)</span>
            </div>
        </div>
    """
    p.hover.tooltips = TOOLTIPS

    p.add_tools(TapTool())
    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url='@image_url')
    wheeltool = p.select(type=WheelZoomTool)
    show_interactive(p)
    #umap.plot.show(p)
#     fig, ax = plt.subplots()
#     slide_sets = [[]] * len(names)
# # [i]nterval_[s]tart, [e]nd
#     for slide_number,(i_s,i_e) in enumerate(zip(slices, slices[1:])):
#         ax.scatter(umap_projection[i_s:i_e, 0], umap_projection[i_s:i_e, 1], label = f"Slide {names[slide_number]}", alpha=.5)
#         slide_sets[slide_number] = umap_projection[i_s:i_e]
#     for i, ss in enumerate(slide_sets):
#         savetxt(os.path.join(args.out_dir, '{}.csv'.format(names[i])), ss, delimiter=',')
#     ax.legend()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # only one of the below, not both
#     #plt.show()
#     plt_dst = os.path.join(args.out_dir, 'cluster', 'umap_output.png')
#     ensure_dir_exists(plt_dst)
#     plt.savefig(plt_dst)

if __name__ == "__main__":
    clinical = pd.read_csv(args.clinical_path, sep='\t')
    clinical = clinical.set_index('case_submitter_id')

    features = pickle.load(open(args.embeddings_path, 'rb'))

    cluster_dst = os.path.join(args.out_dir, 'cluster', f'gmm_{args.n_cluster}.pkl')
    if os.path.exists(cluster_dst):
        cluster = pickle.load(open(cluster_dst, 'rb'))
    else:
        cluster = GaussianMixture(n_components=args.n_cluster, random_state=42).fit([item[0] for sublist in features.values() for item in sublist])
        ensure_dir_exists(cluster_dst)
        pickle.dump(cluster, open(cluster_dst, 'wb'))

    keys_sorted = list(sorted(features.keys()))
    print ("There are {} images in the dataset".format(len(keys_sorted)))
    for i in range(20, len(keys_sorted), 5):
        umap_slice(keys_sorted[i:i+15], features, cluster, clinical, args.out_dir)
        break

    #umap_slice(['TCGA-21-5787-01A-01-TS1'], features, cluster, args.out_dir)
    #umap_slice(['TCGA-43-8115-01A-01-BS1', 'TCGA-34-8456-01A-01-BS1', 'TCGA-68-A59J-01A-02-TSB'], features, cluster, args.out_dir)
# umap_slice(train_features)
