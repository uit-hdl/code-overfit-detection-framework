#!/usr/bin/env -S python -m IPython

import argparse
import pandas as pd
from bokeh.plotting import show as show_interactive
from bokeh.models import ColumnDataSource, OpenURL, TapTool
import operator
import os
import pickle
from itertools import accumulate

import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.plot
from numpy import savetxt


parser = argparse.ArgumentParser(description='Get cluster features')
reducer = umap.UMAP(random_state=42)

parser.add_argument('--embeddings_path', default='./', type=str, help="location of embedding pkl from feature_extraction.py")
parser.add_argument('--cluster_type', default='gmm', type=str)
parser.add_argument('--n_cluster', default=50, type=int)
parser.add_argument('--out_dir', default='./out', type=str)

args = parser.parse_args()

#cluster = GaussianMixture(n_components=args.n_cluster).fit(train_features_flattened)
# pickle.dump(cluster, open(args.out_dir + '/gmm_{}.pkl'.format(args.n_cluster), 'wb'))

def umap_slice(names, features):
    values = [[x[0] for x in features[name]] for name in names]
    tile_names = [[x[1] for x in features[name]] for name in names]
    tile_names = ["file:///" + x for x in np.concatenate(tile_names, axis=0)]
    features_flattened = np.concatenate(values, axis=0)
    umap_projection = reducer.fit_transform(features_flattened)
    mapper = reducer.fit(features_flattened)
    slices = list(accumulate([0] + [len(y) for y in values], operator.add))
    names_labels = [[names[i]] * (i_e - i_s) for i,(i_s,i_e) in enumerate(zip(slices, slices[1:]))]
    names_labels = [item for sublist in names_labels for item in sublist]

    hover_data = pd.DataFrame({'index': np.arange(len(features_flattened)),
                               'image_url': tile_names,
                               'label': names_labels})
    p = umap.plot.interactive(mapper, labels=names_labels, hover_data=hover_data, point_size=7)
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@image_url" height="150" alt="@label" width="150"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@label</span>
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
#     plt.savefig(os.path.join('.', 'umap_output.png'))

if __name__ == "__main__":
    features = pickle.load(open(args.embeddings_path, 'rb'))
    keys_sorted = list(sorted(features.keys()))
    print ("There are {} images in the dataset".format(len(keys_sorted)))
    umap_slice(keys_sorted[:5], features)
    #umap_slice(['TCGA-43-8115-01A-01-BS1', 'TCGA-34-8456-01A-01-BS1', 'TCGA-68-A59J-01A-02-TSB'], features)
# umap_slice(train_features)
