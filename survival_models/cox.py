import sys
sys.path.append('../')
import pickle
from pathlib import Path
import csv
# import  utils
import os
import numpy as np
from survival_models import utils
from sksurv.linear_model import CoxPHSurvivalAnalysis
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Cox-PH survival models')

parser.add_argument('--embeddings_dir', default='out/MoCo/lung_scc/embeddings/', type=str)
parser.add_argument('--cluster_path', default='out/cluster/gmm_50.pkl', type=str)
parser.add_argument('--normalize', default='mean', type=str)
parser.add_argument('--out_dir', default='./out/survival', type=str, help='path to save survival model output')


def ensure_dir_exists(path):
    dest_dir = os.path.dirname(path)
    if not os.path.exists(dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        print(f"mkdir: '{dest_dir}'")

args = parser.parse_args()

cluster = pickle.load(open(args.cluster_path, 'rb'))
cluster_method = type(cluster).__name__
if cluster_method == 'GaussianMixture':
    n_clusters = len(cluster.weights_)
else:
    n_clusters = cluster.n_clusters

train_data, val_data, test_data = utils.load_data(args.embeddings_dir, args.cluster_path, normalize=args.normalize)

train_df, val_df, test_df = utils.preprocess_data(train_data, val_data, test_data, n_clusters)
# if data_source == 'TCGA':
# test_df = test_df.loc[test_df['tcga_flag']==1.0]
# elif data_source =='CPTAC':
#     test_df = test_df.loc[test_df['tcga_flag']==0.0]
# train_df, val_df, test_df = train_df.drop(columns=['tcga_flag']), val_df.drop(columns=['tcga_flag']), test_df.drop(columns=['tcga_flag'])
y_train = np.array([tuple((bool(row[0]), row[1])) for row in zip(train_df['outcome'], train_df['day'])],
        dtype=[('outcome', 'bool'), ('day', '<f4')])
y_val = np.array([tuple((bool(row[0]), row[1])) for row in zip(val_df['outcome'], val_df['day'])],
        dtype=[('outcome', 'bool'), ('day', '<f4')])
y_test = np.array([tuple((bool(row[0]), row[1])) for row in zip(test_df['outcome'], test_df['day'])],
        dtype=[('outcome', 'bool'), ('day', '<f4')])
alpha_list = [10**i for i in np.linspace(-3,0,10)]
alpha = alpha_list[0]
val_results = []

for i, alpha in enumerate(alpha_list):
    est = CoxPHSurvivalAnalysis(alpha=alpha).fit(train_df.drop(columns=['outcome','day']), y_train)
    # val_metrics = utils.get_metrics(train_df, val_df, est)
    _, _, val_c_index = utils.get_metrics(train_df, val_df, est)
    val_results.append(val_c_index)

alpha = alpha_list[np.argmax(val_results)]
est = CoxPHSurvivalAnalysis(alpha=alpha).fit(train_df.drop(columns=['outcome','day']), y_train)
brier2y, brier5y, c_index = utils.get_metrics(train_df, test_df, est)


survival_out_file = os.path.join(args.out_dir, "test_%s_results.csv" % Path(args.cluster_path).stem)
ensure_dir_exists(survival_out_file)
with open(survival_out_file, "w", encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(["Brier2y", "Brier5y", "C-index"])
    writer.writerow([brier2y, brier5y, c_index])
print("Wrote results to %s" % survival_out_file)
# print content of survival_out_file
with open(survival_out_file, "r", encoding='UTF8') as f:
    print(f.read())