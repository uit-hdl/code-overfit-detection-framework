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

#parser.add_argument('--embeddings-path-test', default='out/inceptionv4/checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar/test_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
#parser.add_argument('--embeddings-path-test', default='out/inceptionv4/checkpoint_MoCo_tiles_0020_False_m256_n0_o0_K256.pth.tar/test_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
#parser.add_argument('--embeddings-path-test', default='out/inceptionv4/checkpoint_MoCo_tiles_0020_True_m256_n0_o4_K256.pth.tar/test_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
#parser.add_argument('--embeddings-path-train', default='out/inceptionv4/checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar/train_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
#parser.add_argument('--embeddings-path-val', default='out/inceptionv4/checkpoint_MoCo_tiles_0200_False_m256_n0_o0_K256.pth.tar/val_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
parser.add_argument('--embeddings-path-test', default='out/inceptionv4/checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar/test_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
parser.add_argument('--embeddings-path-train', default='out/inceptionv4/checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar/train_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
parser.add_argument('--embeddings-path-val', default='out/inceptionv4/checkpoint_MoCo_tiles_0200_True_m256_n0_o4_K256.pth.tar/val_tiles_embedding.pkl', type=str, help="location of embedding pkl from feature_extraction.py")
parser.add_argument('--number-of-images', default=30, type=int, help="how many images to sample for the UMAP plot")
parser.add_argument('--histogram-bins', default=20, type=int, help="how many histogram buckets to use in each (x,y) dimension (e.g. 10 means 100 buckets in total)")
parser.add_argument('--clinical-path', default='./annotations/TCGA/clinical.tsv', type=str, help="location of file containing clinical data")
parser.add_argument('--thumbnail-path', default='/Data/TCGA_LUSC/thumbnails', type=str, help="location of directory containing thumbnails")
parser.add_argument('--slide_annotation_file', default=os.path.join('annotations', 'slide_label', 'gdc_sample_sheet.2023-08-14.tsv'), type=str,
                    help='"Sample sheet" from TCGA, see README.md for instructions on how to get sheet')
parser.add_argument('--out-dir', default='./out', type=str)

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
            <span style="font-size: 15px;">Path_m:</span>
            <span style="font-size: 15px;">@path_stage_m</span>
        </div>
        <div>
            <span style="font-size: 15px;">Path_t:</span>
            <span style="font-size: 15px;">@path_stage_t</span>
        </div>
        <div>
            <span style="font-size: 15px;">Resection site</span>
            <span style="font-size: 15px;">@resection</span>
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

def umap_slice(names, features, clinical, slide_annotations):
    values = [[x[0] for x in features[name]] for name in names]
    if not all(values):
        raise RuntimeError("One of the keys did not lead anywhere!")
    tile_names = [[x[1] for x in features[name]] for name in names]
    if len(tile_names[0]) < 10:
        print(f"Skipping {names}, too few tiles")
        return None, None, None, None, None
    file_root = os.path.abspath("/").replace(os.sep, "/")
    tile_names = ["file:///" + file_root + x for x in np.concatenate(tile_names, axis=0)]
    features_flattened = np.concatenate(values, axis=0)
    perplexity_score = math.floor(len(features_flattened) / 100) if len(features_flattened) > 200 else 2
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
    gender_labels = [clinical['gender'][case].iloc[0] for case in case_submitter_ids]
    site_of_resection_labels = [clinical['site_of_resection_or_biopsy'][case].iloc[0] for case in case_submitter_ids]
    path_stage_t = [clinical['ajcc_pathologic_t'][case].iloc[0] for case in case_submitter_ids]
    path_stage_m = [clinical['ajcc_pathologic_m'][case].iloc[0] for case in case_submitter_ids]
    race_labels = [clinical['race'][case].iloc[0] for case in case_submitter_ids]
    tissue_type = list(map(lambda l: slide_annotations[os.path.basename(os.path.dirname(l))], tile_names))

    # If you want to see what the codes refer to https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes
    institution_labels = [case.split("-")[1] for case in case_submitter_ids]
    patient_labels = [case.split("-")[2] for case in case_submitter_ids]

    data = pd.DataFrame({
        'index': np.arange(len(features_flattened)),
        'gender': gender_labels,
        'race': race_labels,
        'institution': institution_labels,
        'resection': site_of_resection_labels,
        'path_stage_m': path_stage_m,
        'path_stage_t': path_stage_t,
        'tissue_type': tissue_type,
        'slide': names_labels,
        'patient': patient_labels,
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

def main(clinical_path, embeddings_path_test, embeddings_path_val, embeddings_path_train, thumbnail_path, histogram_bins, number_of_images, out_dir):
    clinical = pd.read_csv(clinical_path, sep='\t')
    clinical = clinical.set_index('case_submitter_id')
    slide_annotations = pd.read_csv(args.slide_annotation_file, sep='\t', header=0)
    slide_annotations["my_slide_id"] = slide_annotations["File Name"].map(lambda s: s.split(".")[0])
    # Generate a dictionary using "my_slide_id" as key and "Sample Type" as value
    slide_annotations = slide_annotations.set_index("my_slide_id")
    slide_annotations = slide_annotations["Sample Type"].to_dict()

    print(f"Loading embeddings from {embeddings_path_test}")
    features = pickle.load(open(embeddings_path_test, 'rb'))

    # TODO: experimental
    features_val = pickle.load(open(embeddings_path_val, 'rb'))
    features_train = pickle.load(open(embeddings_path_train, 'rb'))
    features.update(features_val)
    features.update(features_train)
    #features = pickle.load(open(embeddings_path_train, 'rb'))

    keys_sorted = list(sorted(features.keys()))
    cutoff = 350
    splits = [0]
    while splits[-1] < len(keys_sorted):
        i = splits[-1] + cutoff
        if i >= len(keys_sorted):
            splits.append(len(keys_sorted))
            break
        inst_at_split = keys_sorted[i].split("-")[1]
        next_inst = inst_at_split
        while inst_at_split == next_inst:
            i += 1
            next_inst = keys_sorted[i].split("-")[1]
        splits.append(i)

    # basename of src (Data) directory
    bsrc=os.path.basename(os.path.dirname(args.embeddings_path_test)).split("_")[2]

    skewers = pd.DataFrame(columns=["slide", "skewness"])
    for i,j  in zip(splits, splits[1:]):
    #for i, j in zip(range(len(keys_sorted)), range(1, len(keys_sorted) + 1)):
        keys_chosen = keys_sorted[i:j]
        number_of_images = j-i
        out_html = os.path.join(out_dir, f"web_{bsrc}", f"condssl_out_{i}_{number_of_images}_{histogram_bins}.html")
        ensure_dir_exists(out_html)
        #if os.path.exists(out_html):
            #print(f"not regenerating {out_html}: file exists")
            #continue
        print ("There are {} slides in the dataset: using {} in analysis...".format(len(keys_sorted), number_of_images))
        pickle_out = os.path.join(args.out_dir, f"tmp_pickle_{len(keys_chosen)}.pkl")
        if os.path.exists(pickle_out) and 1 == 0:
            d = pickle.load(open(pickle_out, 'rb'))
            mapper, data, knn, knc, cpd = d["mapper"], d["data"], d["knn"], d["knc"], d["cpd"]
        else:
            mapper, data, knn, knc, cpd = umap_slice(keys_chosen, features, clinical, slide_annotations)
            if mapper is None:
                continue
            #pickle_obj = {"mapper": mapper, "data": data, "knn": knn, "knc": knc, "cpd": cpd}
            #pickle.dump(pickle_obj, open(pickle_out, 'wb'), protocol=4)

        umap_plots = []
        for key,title in [
            ('slide', "Slide"),
            ('patient', 'Patient'),
            ('institution', "Institution"),
            ('race', "Race"),
            ('gender', "Gender"),
            ('resection', "Resection Site"),
            ('tissue_type', "Tissue Type"),
            ('path_stage_t', "Pathological Stage T"),
            ('path_stage_m', "Pathological Stage M")
        ]:
            unique_labels = np.unique(data[key])
            title += (" ({} unique colors)".format(len(unique_labels)))
            plot = UmapPlot(mapper, data, key, title, unique_labels, histogram_bins)
            overlaps, mean_overlap, skewness = compute_histograms_overlap(plot.plot_data, key, unique_labels, histogram_bins)
            plot.set_overlaps(overlaps, mean_overlap)
            # sort skewness by value, reversed
            #sorted_skewness = sorted(skewness.items(), key=lambda item: item[1], reverse=True)
            # print 10% of items from sorted_skewness
            #skewness = sorted_skewness[:math.ceil(len(sorted_skewness) * 0.1)]
            plot.set_skewness(skewness)
            skewers.loc[len(skewers.index)] = [keys_chosen[0], skewness]
            umap_plots.append(plot)

        inst_umap_overlap = str(100*list(filter(lambda x: x.data_key == "institution", umap_plots))[0].mean_overlap)[:2]
        out_html = out_html.replace(".html", "_{}.html".format(inst_umap_overlap))

        viz_data(mapper, data, keys_chosen, knn, knc, cpd, thumbnail_path, out_html, umap_plots)
    skewers.to_csv(os.path.join(out_dir, "skewness.csv"))

if __name__ == "__main__":
    args = parser.parse_args()
    # TODO: update histogram ticks to be sideways so that they show and not get crunched for large values
    # TODO: would be sick if I updated the histograms on click-select of group
    # TODO: should highlight best and worst results in the table
    main(args.clinical_path, args.embeddings_path_test, args.embeddings_path_val, args.embeddings_path_train, args.thumbnail_path, args.histogram_bins, args.number_of_images, args.out_dir)

    #keys_randomized = random.sample(keys_sorted, len(keys_sorted))
