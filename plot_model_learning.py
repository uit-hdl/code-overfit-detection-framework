import numpy as np
import pandas as pd
from bokeh.plotting import save, figure, show
from bokeh.layouts import layout
import glob
import sys
def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

def get_model_params(s):
    s_digit = list(map(only_numerics, s.split("_")))
    s_letter = s.split("_")

    n = int(s_digit[7])
    m = int(s_digit[6])
    o = int(s_digit[8])
    K = int(s_digit[9])
    c = s_letter[5]
    return (s_letter[1].split("/")[-1], "MoCo c: {} n: {} m: {} o: {} K: {}".format(c, n, m, o, K))

dir=sys.argv[1]
# read each csv-file in dir into a pandas dataframe
files = glob.glob(dir + "/*.csv")
dfs = [(get_model_params(f), pd.read_csv(f)) for f in files]
plots = []
for (metric, title), df in dfs:
    p = figure(height=750, title="{}: {}".format(metric, title))
    p.line(df['epoch'], df[metric], legend_label=metric[0].upper() + metric[1:], line_color="blue")
    plots.append([p])

gp = layout(plots)
#show(gp)
