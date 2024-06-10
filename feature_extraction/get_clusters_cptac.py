#!/usr/bin/env -S python -m IPython

import argparse
import operator
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from itertools import accumulate

import numpy as np
import pandas as pd
import scipy.stats
import umap
from bokeh.layouts import layout
from bokeh.models import OpenURL, TapTool, ColumnDataSource
from bokeh.models.widgets import Div, DataTable, TableColumn
from bokeh.plotting import output_file, save, figure
from scipy.spatial import distance
from scipy.stats import spearmanr, skew
from sklearn.neighbors import NearestNeighbors

import umap_plot

sys.path.append('./')
from misc.global_util import ensure_dir_exists


parser = argparse.ArgumentParser(description='Get cluster features')

parser.add_argument('--embeddings-path-test', default='out/inceptionv4/checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar/cptac_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
parser.add_argument('--number-of-images', default=2999, type=int, help="how many images to sample for the UMAP plot")
parser.add_argument('--histogram-bins', default=20, type=int, help="how many histogram buckets to use in each (x,y) dimension (e.g. 10 means 100 buckets in total)")
parser.add_argument('--thumbnail-path', default='/Data/TCGA_LUSC/thumbnails', type=str, help="location of directory containing thumbnails")
parser.add_argument('--out-dir', default='./out', type=str)
parser.add_argument('--slide_annotation_file', default=os.path.join('annotations', 'CPTAC', 'slide.tsv'), type=str,
                    help='"Slide sheet", containing sample information, see README.md for instructions on how to get sheet')
parser.add_argument('--sample_annotation_file', default=os.path.join('annotations', 'CPTAC', 'sample.tsv'), type=str,
                    help='"Slide sheet", containing sample information, see README.md for instructions on how to get sheet')

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

def compute_histograms_overlap(plot_data, data_key, unique_labels, no_bins):
    _, edges = np.histogram(plot_data[["x", "y"]], bins=no_bins**2)

    counts = []
    skewness = {}
    for l in unique_labels:
        histo_count, _ = np.histogram(plot_data[plot_data[data_key] == l][["x", "y"]], bins=edges)
        non_zero_bins = list(filter(lambda x: x != 0, histo_count))
        skewness[l] = skew(non_zero_bins)
        counts.append((l, histo_count))
    skewness["all"] = np.mean(list(skewness.values()))
    overlaps = defaultdict(lambda: defaultdict(list))
    all_counts = np.sum([x[1] for x in counts])
    for (label_left,count_left) in counts:
        #cc = np.array([count_left, all_counts - count_left])
        #overlaps[label_left]["all"] = np.mean(np.divide(cc.min(axis=0), cc.max(axis=0), where=cc.min(axis=0)!=0))
        for label_right,count_right in counts:
            if label_left == label_right:
                continue
            #cc = np.array([count_left, count_right])
            #division = np.divide(cc.min(axis=0), cc.max(axis=0))
            # problem with line below: doesn't respect where there are cells in one line, but not the other
            #overlaps[label_left][label_right] = np.mean(np.divide(cc.min(axis=0), cc.max(axis=0), where=cc.min(axis=0)!=0))
            # problem with below: doesn't take into account that there is a different number of tiles altogether
            #overlaps[label_left][label_right] = np.mean(np.nan_to_num(division[division != 0]))
            #pearson, pearson_p = scipy.stats.pearsonr(count_left, count_right)
            #spearman = scipy.stats.spearmanr(count_left, count_right)
            #kendalltau = scipy.stats.kendalltau(count_left, count_right)
            overlaps[label_left][label_right] = scipy.stats.kendalltau(count_left, count_right).correlation

        #overlaps[label_left]["all"] = scipy.stats.kendalltau(count_left, all_counts - count_left).correlation
        if len(overlaps[label_left].keys()) == 0:
            overlaps[label_left]["all"] = 0.0
        else:
            overlaps[label_left]["all"] = np.mean(list(overlaps[label_left].values()))

    mean_overlap = np.mean([x["all"] for x in overlaps.values()])
    return overlaps, mean_overlap, skewness

def umap_slice(names, features, slide_annotations):
    values = [[x[0] for x in features[name]] for name in names]
    if not all(values):
        raise RuntimeError("One of the keys did not lead anywhere!")
    tile_names = [[x[1] for x in features[name]] for name in names]
    file_root = os.path.abspath("/").replace(os.sep, "/")
    tile_names = ["file:///" + file_root + x for x in np.concatenate(tile_names, axis=0)]
    features_flattened = np.concatenate(values, axis=0)
    perplexity_score = math.floor(len(features_flattened) / 100)
    reducer = umap.UMAP(random_state=42, n_neighbors=perplexity_score, min_dist=1.0)
    umap_projection = reducer.fit_transform(features_flattened)
    #mapper = reducer.fit(features_flattened)

    # #1
    knn_fractions = None #compute_knn(features_flattened, umap_projection, k=10)
    knc_fractions = None # compute_knc(features_flattened, umap_projection, cluster_labels, k=10)
    cpd = None # compute_cpd(features_flattened, umap_projection, sample_size=1000)

    #umap_projection = reducer.fit_transform(features_flattened)
    slices = list(accumulate([0] + [len(y) for y in values], operator.add))
    names_labels = [[names[i]] * (i_e - i_s) for i,(i_s,i_e) in enumerate(zip(slices, slices[1:]))]
    names_labels = [item for sublist in names_labels for item in sublist]
    case_submitter_ids = ["-".join(name.split("-")[:3]) for name in names_labels]
    tissue_type = list(map(lambda l: slide_annotations.loc[l]['sample_type'], names_labels))

    patient_labels = [case.split("-")[1] for case in case_submitter_ids]

    data = pd.DataFrame({
        'index': np.arange(len(features_flattened)),
        'slide': names_labels,
        'patient': patient_labels,
        'tissue_type': tissue_type,
        'image_url': tile_names,
    })
    return umap_projection, data, knn_fractions, knc_fractions, cpd

def plot_umap_scatter(umap_projection, data, data_key, title, no_bins):
    labels = data[data_key]
    unique_labels = np.unique(labels)

    color_key_cmap = "tab20"
    if len(unique_labels) > 20:
        color_key_cmap = "Spectral"
    p, plot_data, color_key, text_search, multibox_input, distribution_plot = \
        umap_plot.interactive(umap_projection, width=1000, height=1000, color_key_cmap=color_key_cmap, labels=labels,
                              hover_data=data, point_size=3, hover_tips=TOOLTIPS, title=title,
                              interactive_text_search=True, interactive_text_search_columns=[data_key])

    plot_href, plot_vref = compute_scatter_histograms(umap_projection, labels, plot_data, data_key, no_bins)

    ph = figure(toolbar_location=None, width=p.width, height=200, min_border=10, min_border_left=50, y_axis_location="right", x_range=plot_href["hedges"], y_axis_label="no. of points")
    ph.yaxis.axis_label_text_align = "left"
    ph.xgrid.grid_line_color = None
    ph.yaxis.major_label_orientation = np.pi / 4
    ph.xaxis.visible = False
    ph.vbar_stack(list(map(str, unique_labels)), source=plot_href, x='hedges', color=color_key)

    pv = figure(toolbar_location=None, width=200, height=p.height, min_border=10, y_axis_location="right", y_range=plot_vref["vedges"], x_axis_label="no. of points")
    ph.xaxis.axis_label_text_align = "left"
    pv.ygrid.grid_line_color = None
    pv.yaxis.visible = False
    pv.background_fill_color = "#fafafa"

    pv.hbar_stack(list(map(str, unique_labels)), source=plot_vref, y='vedges', color=color_key)

    return p, pv, ph, plot_data, text_search, multibox_input, distribution_plot

def compute_scatter_histograms(embedding, labels, plot_data, data_key, no_bins):
    unique_labels = np.unique(labels)

    _, hedges = np.histogram(embedding[:, 0], bins=no_bins)

    bucket_href = list(map(str, hedges))[:-1]
    plot_href = {'hedges': bucket_href}
    for l in unique_labels:
        plot_href[str(l)] = np.histogram(plot_data[plot_data[data_key] == l]["x"], bins=hedges)[0]

    _, vedges = np.histogram(embedding[:, 1], bins=no_bins)
    bucket_vref = list(map(str, vedges))[:-1]
    plot_vref = {'vedges': bucket_vref}
    for l in unique_labels:
        plot_vref[str(l)] = np.histogram(plot_data[plot_data[data_key] == l]["y"], bins=vedges)[0]

    return plot_href, plot_vref

class UmapPlot:
    def __init__(self, mapper, data, data_key, title, unique_labels, no_bins):
        self.overlaps = None
        self.mean_overlap = 0.0
        p, pv, ph, plot_data, text_search, multibox_input, distribution_plot = plot_umap_scatter(mapper, data, data_key, title, no_bins)
        self.p = p
        self.pv = pv
        self.ph = ph
        self.plot_data = plot_data
        self.title = title
        self.data_key = data_key
        self.unique_labels = unique_labels
        self.text_search = text_search
        self.multibox_input = multibox_input
        self.distribution_plot = distribution_plot
        self.skewness = None

        if len(self.unique_labels) > 5:
            p.legend.visible = False

        self.p.legend.location = "top_left"
    def set_overlaps(self, overlaps, mean_overlap):
        ordering = {k: i for i, k in enumerate(overlaps.keys())}
        ordering["all"] = len(self.unique_labels)

        counts = []
        # convert the overlaps into a 2d-matrix
        for key,d in overlaps.items():
            dst_array = [0] * (len(self.unique_labels) + 1)
            for key2,v in d.items():
                dst_array[ordering[key2]] = v
            counts.append(dst_array)
        self.overlaps = counts
        self.mean_overlap = mean_overlap

    def set_skewness(self, skewness):
        self.skewness = skewness

    def render_to_bokeh(self):
        return [[self.p, self.ph, self.distribution_plot], [self.pv], [self.text_search, self.multibox_input]]


def viz_data(mapper, data, names, knn, knc, cpd, thumbnail_path, out_html, umap_plots):
    output_file(out_html, title="Conditional SSL UMAP")
    print("Outputting to {}".format(out_html))
    stat_box = Div(text="""<p style="font-size: 500%">CPD: {:.4f} (p {:.1f})<br />KNC mean {:.4f}<br />KNN mean {:.4f}<br /></p>"""
                   .format(cpd[0] if cpd else 0, cpd[1] if cpd else 0, np.mean(knc) if knc else 0, np.mean(knn) if knn else 0))

    image_links = """<style>
    .top-left {
        position: absolute;
        bottom: 8px;
        left: 16px;
    }
    .gallery {
          --s: 400px; /* control the size */
          display: grid;
          gap: 10px; /* control the gap */
          grid: auto-flow var(--s)/repeat(3,var(--s));
          place-items: center;
          margin: calc(var(--s)/4);
    }
    .gallery > img {
      width: 100%;
      aspect-ratio: 1;
    }
    </style><div class="gallery">"""
    #thumbnails = set(map(lambda s: os.path.dirname(s).replace("preprocessed/by_class/lung_scc", "thumbnails") + ".png", tile_names))
    for n in names:
        tp = os.path.join(thumbnail_path, n + ".png")
        tp = f"file:///{tp}"
        image_links+=f"""<img src="{tp}" title="{n}"/>"""
    image_links += "</div>"
    if len(names) > 9:
        image_links = ""
    image_thumbnail = Div(text=image_links)

    data_tables = []
    for plot in umap_plots:
        data = dict(keys=list(map(str, plot.unique_labels)) + ["all"])
        columns = []
        for i,overlap in enumerate(plot.overlaps):
            data[f"counts_{i+1}"] = list(map(lambda f: f"{f:.2}" if f != 0.0 else "-", overlap))
            columns.append(TableColumn(field=f"counts_{i+1}", title=str(plot.unique_labels[i])))
        source = ColumnDataSource(data)
        columns = [ TableColumn(field="keys", title=plot.title)] + columns
        data_tables.append(DataTable(source=source, columns=columns, width=800, height=280, index_position=None))

    mean_overlaps = "Mean overlaps (overall): <br />"
    for plot in umap_plots:
        mean_overlaps += """{}: {:.2%}<br />""".format(plot.title, plot.mean_overlap)

    mean_overlaps_box = Div(text="""<p style="font-size: 300%">{}</p>""".format(mean_overlaps))


    data_tables_skewness = []
    for plot in umap_plots:
        d = plot.skewness
        data = {'ids': list(d.keys()), 'values': list(d.values())}
        source = ColumnDataSource(data)
        columns = [
            TableColumn(field="ids", title=plot.title),
            TableColumn(field="values", title="Skewness"),
        ]
        data_tables_skewness.append(DataTable(source=source, columns=columns, width=800, height=280))

    #gp = layout([[image_thumbnail], [p1, pv1], [ph1], [p3, pv3], [ph3], [stat_box], [data_table]])
    components = [plot.render_to_bokeh() for plot in umap_plots]
    gp = layout([[image_thumbnail]] + components +  [[stat_box], [data_tables] + [[mean_overlaps_box]], [data_tables_skewness]])
    tt = TapTool()
    tt.callback = OpenURL(url="@image_url")
    umap_plots[0].p.tools.append(tt)
    #gp.toolbar.tools.append(tt)

    save(gp)

def main(embeddings_path_test, thumbnail_path, histogram_bins, number_of_images, out_dir):
    slide_annotations = pd.read_csv(args.slide_annotation_file, sep='\t', header=0)
    slide_annotations = slide_annotations[['slide_submitter_id', 'sample_id']]

    sample_annotations = pd.read_csv(args.sample_annotation_file, sep='\t', header=0)
    sample_annotations = sample_annotations[['sample_id', 'sample_type']]

    df = slide_annotations.merge(sample_annotations, on='sample_id')
    df = df.set_index('slide_submitter_id')

    for label,embeddings_path_test in [("cptac", embeddings_path_test)]:
        print(f"Loading embeddings from {embeddings_path_test}")
        features = pickle.load(open(embeddings_path_test, 'rb'))

        keys_sorted = list(sorted(features.keys()))
        keys_chosen = keys_sorted
        number_of_images = len(keys_chosen)
        print ("There are {} slides in the dataset: using {} in analysis...".format(len(keys_sorted), number_of_images))
        pickle_out = os.path.join(args.out_dir, f"tmp_pickle_{len(keys_chosen)}.pkl")
        if os.path.exists(pickle_out) and 1 == 0:
            d = pickle.load(open(pickle_out, 'rb'))
            mapper, data, knn, knc, cpd = d["mapper"], d["data"], d["knn"], d["knc"], d["cpd"]
        else:
            mapper, data, knn, knc, cpd = umap_slice(keys_chosen, features, df)
            #pickle_obj = {"mapper": mapper, "data": data, "knn": knn, "knc": knc, "cpd": cpd}
            #pickle.dump(pickle_obj, open(pickle_out, 'wb'), protocol=4)


        umap_plots = []
        for key,title in [
            ('slide', "Slide"),
            ('patient', 'Patient'),
            ('tissue_type', "Tissue Type"),
        ]:
            unique_labels = np.unique(data[key])
            title += (" ({} unique colors)".format(len(unique_labels)))
            plot = UmapPlot(mapper, data, key, title, unique_labels, histogram_bins)
            overlaps, mean_overlap, skewness = compute_histograms_overlap(plot.plot_data, key, unique_labels, histogram_bins)
            plot.set_overlaps(overlaps, mean_overlap)
            plot.set_skewness(skewness)
            umap_plots.append(plot)

        out_html = os.path.join(out_dir, "web", "cptac", f"condssl_cptac_out_{label}_{number_of_images}_{histogram_bins}.html")
        ensure_dir_exists(out_html)

        viz_data(mapper, data, keys_chosen, knn, knc, cpd, thumbnail_path, out_html, umap_plots)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.embeddings_path_test, args.thumbnail_path, args.histogram_bins, args.number_of_images, args.out_dir)
