#!/usr/bin/env -S python -m IPython

import argparse
import operator
import random
from scipy.stats import spearmanr

import pandas
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import os
import pickle
from itertools import accumulate
from pathlib import Path
from bokeh.models import OpenURL, TapTool
from bokeh.models.widgets import Paragraph

import numpy as np
import pandas as pd
import umap.plot
from bokeh.layouts import gridplot
from bokeh.plotting import output_file, save, figure
from bokeh.models import Whisker
from sklearn.mixture import GaussianMixture

import umap_plot


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
            <span style="font-size: 15px;">Institution:</span>
            <span style="font-size: 15px;">@institution</span>
        </div>
        <div>
            <span style="font-size: 15px;">Race:</span>
            <span style="font-size: 15px;">@race</span>
        </div>
        <div>
            <span style="font-size: 15px;">Location</span>
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
    </div>
"""

def compute_knn(high_dimensional_points, low_dimensional_points, k=10):
    nbrs_high = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(high_dimensional_points)
    nbrs_low = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(low_dimensional_points)

    distances_high, indices_high = nbrs_high.kneighbors(high_dimensional_points)
    distances_low, indices_low = nbrs_low.kneighbors(low_dimensional_points)
    # Calculate how many elements in b that is in a
    knn_frac = lambda s: len(set(s[0]).intersection(s[1])) / k
    knn_fractions = list(map(knn_frac, zip(indices_low, indices_high)))

    return knn_fractions

def compute_knc(high_dimensional_points, low_dimensional_points, cluster_labels, k=10):
    nbrs_high = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(high_dimensional_points)
    nbrs_low = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(low_dimensional_points)

    distances_high, indices_high = nbrs_high.kneighbors(high_dimensional_points)
    distances_low, indices_low = nbrs_low.kneighbors(low_dimensional_points)
    cluster_map = lambda x: cluster_labels[x]
    class_mapping_high = [list(map(cluster_map, li)) for li in indices_high]
    class_mapping_low = [list(map(cluster_map, li)) for li in indices_low]

    # Calculate how many elements in b that is in a
    knn_frac = lambda s: len(set(s[0]).intersection(s[1])) / k
    knn_fractions = list(map(knn_frac, zip(class_mapping_high, class_mapping_low)))

    return knn_fractions

def compute_cpd(high_dimensional_points, low_dimensional_points, sample_size=1000):
    points = random.sample(range(len(high_dimensional_points)), sample_size if len(high_dimensional_points) > sample_size else len(high_dimensional_points))
    dists_high = [0] * (len(points) * (len(points) -1))
    dists_low = [0] * (len(points) * (len(points) - 1))
    index = 0
    for i,p in enumerate(points):
        for j,q in enumerate(points):
            if i == j:
                continue
            dists_high[index] = distance.euclidean(high_dimensional_points[i], high_dimensional_points[j])
            dists_low[index] = distance.euclidean(low_dimensional_points[i], low_dimensional_points[j])
            index += 1
    return spearmanr(dists_high, dists_low)




def umap_slice(names, features, cluster, clinical, out_dir):
    values = [[x[0] for x in features[name]] for name in names]
    cluster_labels = [cluster.predict(y) for y in values]
    cluster_labels = [item for sublist in cluster_labels for item in sublist]
    tile_names = [[x[1] for x in features[name]] for name in names]
    file_root = os.path.abspath("/").replace(os.sep, "/")
    tile_names = ["file:///" + file_root + x for x in np.concatenate(tile_names, axis=0)]
    features_flattened = np.concatenate(values, axis=0)
    umap_projection = reducer.fit_transform(features_flattened)
    mapper = reducer.fit(features_flattened)

    # TODO: #1 make a KNN in high-dimensional space
    # TODO: #2 make a KNN in low-dimensional space
    # TODO: compute how many of the neighbors from #1 that are preserved in #2
    # (The art of using t-SNE for single-cell transcriptomics)

    # #1
    knn_fractions = compute_knn(features_flattened, mapper.embedding_, k=10)
    knc_fractions = compute_knc(features_flattened, mapper.embedding_, cluster_labels, k=10)
    cpd = compute_cpd(features_flattened, mapper.embedding_, sample_size=1000)

    #umap_projection = reducer.fit_transform(features_flattened)
    slices = list(accumulate([0] + [len(y) for y in values], operator.add))
    names_labels = [[names[i]] * (i_e - i_s) for i,(i_s,i_e) in enumerate(zip(slices, slices[1:]))]
    names_labels = [item for sublist in names_labels for item in sublist]
    case_submitter_ids = ["-".join(name.split("-")[:3]) for name in names_labels]
    gender_labels = [clinical['gender'][case][0] for case in case_submitter_ids]
    race_labels = [clinical['race'][case][0] for case in case_submitter_ids]
    # If you want to see what the codes refer to https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes
    institution_labels = [case.split("-")[1] for case in case_submitter_ids]

    hover_data = pd.DataFrame({'index': np.arange(len(features_flattened)),
                               'cluster_id': cluster_labels,
                               'gender': gender_labels,
                               'race': race_labels,
                               'institution': institution_labels,
                               'slide': names_labels,
                               'image_url': tile_names,
                               })

    out_html = os.path.join(out_dir, "web", "condssl_out.html")
    ensure_dir_exists(out_html)
    output_file(out_html, title="Conditional SSL UMAP")
    p1 = umap_plot.interactive(mapper, labels=names_labels, hover_data=hover_data, point_size=7, hover_tips=TOOLTIPS, title="Slide")
    if len(names) > 5: p1.legend.visible = False
    p2 = umap_plot.interactive(mapper, labels=gender_labels, hover_data=hover_data, point_size=7, hover_tips=TOOLTIPS, title="Gender")
    p3 = umap_plot.interactive(mapper, labels=institution_labels, hover_data=hover_data, point_size=7, hover_tips=TOOLTIPS, title="Instiution")
    p4 = umap_plot.interactive(mapper, labels=race_labels, hover_data=hover_data, point_size=7, hover_tips=TOOLTIPS, title="Race")
    p5 = umap_plot.interactive(mapper, labels=cluster_labels, hover_data=hover_data, point_size=7, hover_tips=TOOLTIPS, title="GMM Cluster")
    for plot in [p1, p2, p3, p4, p5]:
        plot.legend.location = "top_left"

    stat_box = Paragraph(text="""CPD: {:.4f} (pvalue {:.4f})<br />KNC mean {:.4f}<br />KNN mean {:.4f}""".format(cpd[0], cpd[1], np.mean(knc_fractions), np.mean(knn_fractions)))

    gp = gridplot([[p1, p3], [p2, p4], [stat_box, p5]])
    #gp = gridplot([[p1]])
    tt = TapTool()
    tt.callback = OpenURL(url="@image_url")
    p1.tools.append(tt)
    #gp.toolbar.tools.append(tt)

    save(gp)
    #show_interactive(gp)
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
    #umap_slice(keys_sorted[8:16], features, cluster, clinical, args.out_dir)

    #keys_chosen = [k for k in keys_sorted if k.split("-")[1] in ["66", "63"]]
    #keys_chosen = [k for k in keys_sorted if k.split("-")[1] in ["94", "63"]]
    #keys_chosen = [k for k in keys_sorted if k.split("-")[1] in ["43", "21"]]
    # TODO: would be sick if I could get a preview of the whole WSI on top in the page
    keys_chosen = [k for k in keys_sorted if k.split("-")[1] in ["96", "94", "58"]]
    umap_slice(keys_chosen, features, cluster, clinical, args.out_dir)

    #keys_randomized = random.sample(keys_sorted, len(keys_sorted))
    #umap_slice(keys_randomized[8:14], features, cluster, clinical, args.out_dir)

    #umap_slice(['TCGA-21-5787-01A-01-TS1'], features, cluster, args.out_dir)
    #umap_slice(['TCGA-43-8115-01A-01-BS1', 'TCGA-34-8456-01A-01-BS1', 'TCGA-68-A59J-01A-02-TSB'], features, cluster, args.out_dir)
# umap_slice(train_features)
