from warnings import warn

import bokeh.plotting as bpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.models import CustomJS, TextInput
from bokeh.models import OpenURL, TapTool, ColumnDataSource
from bokeh.models import MultiSelect
from bokeh.plotting import output_file, save, figure


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]

def _get_embedding(umap_object):
    if hasattr(umap_object, "embedding_"):
        return umap_object.embedding_
    elif hasattr(umap_object, "embedding"):
        return umap_object.embedding
    else:
        raise ValueError("Could not find embedding attribute of umap_object")


def interactive(
        umap_object,
        labels,
        hover_data=None,
        color_key_cmap="Spectral",
        width=800,
        height=800,
        point_size=None,
        interactive_text_search=False,
        interactive_text_search_columns=None,
        interactive_text_search_alpha_contrast=0.75,
        alpha=None,
        hover_tips=None,
        title=None,
):
    """Create an interactive bokeh plot of a UMAP embedding.
    While static plots are useful, sometimes a plot that
    supports interactive zooming, and hover tooltips for
    individual points is much more desireable. This function
    provides a simple interface for creating such plots. The
    result is a bokeh plot that will be displayed in a notebook.

    Note that more complex tooltips etc. will require custom
    code -- this is merely meant to provide fast and easy
    access to interactive plotting.

    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.

    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label.

    hover_data: DataFrame, shape (n_samples, n_tooltip_features)
    (optional, default None)
        A dataframe of tooltip data. Each column of the dataframe
        should be a Series of length ``n_samples`` providing a value
        for each data point. Column names will be used for
        identifying information within the tooltip.

    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    width: int (optional, default 800)
        The desired width of the plot in pixels.

    height: int (optional, default 800)
        The desired height of the plot in pixels

    point_size: int (optional, default None)
        The size of each point marker

    interactive_text_search: bool (optional, default False)
        Whether to include a text search widget above the interactive plot

    interactive_text_search_columns: list (optional, default None)
        Columns of data source to search. Searches labels and hover_data by default.

    interactive_text_search_alpha_contrast: float (optional, default 0.95)
        Alpha value for points matching text search. Alpha value for points
        not matching text search will be 1 - interactive_text_search_alpha_contrast

    alpha: float (optional, default: None)
        The alpha blending value, between 0 (transparent) and 1 (opaque).

    Returns
    -------
    """

    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1 inclusive")

    points = _get_embedding(umap_object)

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    data = pd.DataFrame(_get_embedding(umap_object), columns=("x", "y"))

    data["label"] = labels

    unique_labels = np.unique(labels)
    num_labels = unique_labels.shape[0]
    color_key = _to_hex( plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)) )

    new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
    data["color"] = pd.Series(labels).map(new_color_key)

    colors = "color"

    if not points.shape[0] <= width * height // 10:
        raise ValueError("Too many points to plot interactively, use umap's official lib instead")

    if hover_data is not None:
        tooltip_dict = {}
        for col_name in hover_data:
            data[col_name] = hover_data[col_name]
            tooltip_dict[col_name] = "@{" + col_name + "}"
        tooltips = list(tooltip_dict.items())
    else:
        tooltips = None

    data["alpha"] = alpha if alpha else 1

    # bpl.output_notebook(hide_banner=True) # this doesn't work for non-notebook use
    data_source = bpl.ColumnDataSource(data)

    plot = bpl.figure(
        width=width,
        height=height,
        tooltips=tooltips,
    )
    plot.circle(
        x="x",
        y="y",
        source=data_source,
        legend_group="label",
        color=colors,
        muted_color=colors,
        size=point_size,
        alpha="alpha",
    )

    # Doesn't work, just hides all the points?
    #plot.legend.click_policy="mute"

    if hover_tips is not None:
        plot.hover.tooltips = hover_tips
    if title is not None:
        plot.title = title

    plot.grid.visible = False
    plot.axis.visible = False

    if 1 == 1:
        callback = CustomJS(
            args = dict(
            source=data_source,
            matching_alpha=interactive_text_search_alpha_contrast,
            non_matching_alpha=1 - interactive_text_search_alpha_contrast,
            search_columns=interactive_text_search_columns,
        ),
        code = """
        var data = source.data;
        var text_search = cb_obj.value;

        var search_columns_dict = {}
        for (var col in search_columns){
            search_columns_dict[col] = search_columns[col]
        }
        // Loop over columns and values
        // If there is no match for any column for a given row, change the alpha value
        var string_match = false;
        for (var i = 0; i < data.x.length; i++) {
            string_match = false
            for (var j in search_columns_dict) {
                if (cb_obj.value.includes(String(data[search_columns_dict[j]][i]))) {
                    string_match = true
                }
            }
            if (string_match){
                data['alpha'][i] = 1;//matching_alpha; // FIXME: matching_alpha doesn't work, just turns off the color
            }else{
                data['alpha'][i] = non_matching_alpha;
            }
        }
        source.change.emit();
        """,
        )
        # if you want to improve: https://docs.bokeh.org/en/2.4.0/docs/reference/models/widgets/inputs.html#bokeh.models.MultiSelect.js_link
        multibox_input = MultiSelect(options=list(map(str, unique_labels)), size=50)
        multibox_input.js_on_change("value", callback)

    if interactive_text_search:
        text_input = TextInput(value="", title="Search:")

        if interactive_text_search_columns is None:
            interactive_text_search_columns = []
            if hover_data is not None:
                interactive_text_search_columns.extend(hover_data.columns)
            if labels is not None:
                interactive_text_search_columns.append("label")

        if len(interactive_text_search_columns) == 0:
            warn(
                "interactive_text_search_columns set to True, but no hover_data or labels provided."
                "Please provide hover_data or labels to use interactive text search."
            )

        else:
            callback = CustomJS(
                args=dict(
                    source=data_source,
                    matching_alpha=interactive_text_search_alpha_contrast,
                    non_matching_alpha=1 - interactive_text_search_alpha_contrast,
                    search_columns=interactive_text_search_columns,
                ),
                code="""
                var data = source.data;
                var text_search = cb_obj.value;

                var search_columns_dict = {}
                for (var col in search_columns){
                    search_columns_dict[col] = search_columns[col]
                }
                // Loop over columns and values
                // If there is no match for any column for a given row, change the alpha value
                var string_match = false;
                for (var i = 0; i < data.x.length; i++) {
                    string_match = false
                    for (var j in search_columns_dict) {
                        if (String(data[search_columns_dict[j]][i]).includes(text_search) ) {
                            string_match = true
                        }
                    }
                    if (string_match){
                        data['alpha'][i] = 1;//matching_alpha; // FIXME: matching_alpha doesn't work, just turns off the color
                    }else{
                        data['alpha'][i] = non_matching_alpha;
                    }
                }
                source.change.emit();
            """,
            )

            text_input.js_on_change("value", callback)
    else:
        text_input = None
        multibox_input = None

    counts = [np.sum(labels == l) for l in unique_labels]
    unique_labels = list(map(str, unique_labels))
    source = ColumnDataSource(data=dict(item=unique_labels, counts=counts, color=color_key))
    distribution_plot = figure(toolbar_location=None, title="Distribution plot (number of points overall)", tools="hover", tooltips=[("Count", "@counts")], width=600, height=200, x_range=unique_labels)
    distribution_plot.vbar(source=source, x='item', top='counts', width=0.9, color='color')
    #distribution_plot.vbar(source=source, x=unique_labels, top='counts', width=0.9, color='color', legend_field='item')
    distribution_plot.y_range.start = 0
    #distribution_plot.legend.visible = False
    if len(unique_labels) > 10:
        distribution_plot.xaxis.major_label_orientation = 1.2
    #new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}


    return plot, data, color_key, text_input, multibox_input, distribution_plot

