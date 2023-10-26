#!/usr/bin/env python

import numpy as np
import pandas as pd
from bokeh.plotting import save, figure

ph = figure(toolbar_location=None, height=200, min_border=10, min_border_left=50, y_axis_location="right")
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi / 4
ph.background_fill_color = "#fafafa"

labels = pd.Series(['green', 'blue', 'green'])
points = np.array([[1,1],[2,2],[3,3]])

data = pd.DataFrame(points, columns=("x", "y"))
data["label"] = labels
unique_labels = np.unique(labels)
num_labels = unique_labels.shape[0]
hhist, hedges = np.histogram(points[:, 0], bins=10)

ph.vbar_stack(range(10), source=data, x='')

save(ph, "out.html")
    #ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
