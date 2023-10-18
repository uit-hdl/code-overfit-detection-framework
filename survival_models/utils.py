import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, concordance_index_censored, integrated_brier_score
import pickle
import os
from collections import defaultdict
from tqdm import tqdm


def get_metrics(train_df, test_df, est):
    metrics = {}
    y_train = np.array([tuple((bool(row[0]), row[1])) for row in zip(train_df['outcome'], train_df['day'])],
                       dtype=[('outcome', 'bool'), ('day', '<f4')])
    y_test = np.array([tuple((bool(row[0]), row[1])) for row in zip(test_df['outcome'], test_df['day'])],
                      dtype=[('outcome', 'bool'), ('day', '<f4')])
    survs = est.predict_survival_function(test_df.drop(columns=['outcome', 'day']))
    try:
        times = np.arange(4, 10)
        preds = np.asarray([[fn(t) for t in times] for fn in survs])
    except:
        print('no five year records')
    preds = [fn(4) for fn in survs]
    import ipdb; ipdb.set_trace()
    times, score = brier_score(y_train, y_test, preds, 4)
    metrics['Brier_2yr'] = score[0]
    try:
        preds = [fn(10) for fn in survs]
        times, score = brier_score(y_train, y_test, preds, 10)
        metrics['Brier_5yr'] = score[0]
    except:
        print('no five year records')
    preds = est.predict(test_df.drop(columns=['outcome', 'day']))
    # TODO: wtf? how was the line below ever supposed to make sense? from upstream/main
    # metrics['C-index'] = concordance_index_censored(y_train, y_test, preds)[0]
    metrics['C-index'] = concordance_index_censored([x[0] for x in y_test], [x[1] for x in y_test], preds)[0]
    return metrics


def preprocess_data(train_data, val_data, test_data, n_clusters):
    if 'tcga' not in train_data:
        train_df = pd.DataFrame(np.concatenate([train_data['cluster'], np.stack(
            [train_data['recur_day'], train_data['followup_day'], train_data['outcome']]).T], axis=1),
                                columns=['c_{}'.format(i) for i in range(n_clusters)] + ['recur', 'followup', 'outcome'])
    else:
        train_df = pd.DataFrame(np.concatenate([train_data['cluster'], np.stack(
            [train_data['recur_day'], train_data['followup_day'], train_data['outcome']]).T], axis=1),
                                columns=['c_{}'.format(i) for i in range(train_data['cluster'].shape[1])] + ['recur',
                                                                                                             'followup',
                                                                                                             'outcome'])
    train_df = train_df[(train_df['recur'].notna() | train_df['followup'].notna())]
    train_df['day'] = train_df['recur']
    # import ipdb; ipdb.set_trace()
    train_df.loc[train_df['recur'].isna(), 'day'] = train_df.loc[train_df['recur'].isna(), 'followup']
    train_df = train_df.drop(columns=['recur', 'followup'])
    train_df = train_df[train_df['day'] > 0]
    train_df['day'] = train_df['day'] // 180 + 1

    if 'tcga' not in val_data:
        val_df = pd.DataFrame(np.concatenate(
            [val_data['cluster'], np.stack([val_data['recur_day'], val_data['followup_day'], val_data['outcome']]).T],
            axis=1),
                              columns=['c_{}'.format(i) for i in range(n_clusters)] + ['recur', 'followup', 'outcome'])
    else:
        val_df = pd.DataFrame(np.concatenate(
            [val_data['cluster'], np.stack([val_data['recur_day'], val_data['followup_day'], val_data['outcome']]).T],
            axis=1), columns=['c_{}'.format(i) for i in range(val_data['cluster'].shape[1])] + ['recur', 'followup',
                                                                                                'outcome'])
    val_df = val_df[(val_df['recur'].notna() | val_df['followup'].notna())]
    val_df['day'] = val_df['recur']
    val_df.loc[val_df['recur'].isna(), 'day'] = val_df.loc[val_df['recur'].isna(), 'followup']
    val_df = val_df.drop(columns=['recur', 'followup'])
    val_df = val_df[val_df['day'] > 0]
    val_df['day'] = val_df['day'] // 180 + 1

    if 'tcga' not in test_data:
        test_df = pd.DataFrame(np.concatenate([test_data['cluster'], np.stack(
            [test_data['recur_day'], test_data['followup_day'], test_data['outcome']]).T], axis=1),
                               columns=['c_{}'.format(i) for i in range(n_clusters)] + ['recur', 'followup', 'outcome'])
    else:
        test_df = pd.DataFrame(np.concatenate([test_data['cluster'], np.stack(
            [test_data['recur_day'], test_data['followup_day'], test_data['outcome']]).T], axis=1),
                               columns=['c_{}'.format(i) for i in range(test_data['cluster'].shape[1])] + ['recur', 'followup', 'outcome'])

    test_df['day'] = test_df['recur']
    test_df = test_df[(test_df['recur'].notna() | test_df['followup'].notna())]

    test_df.loc[test_df['recur'].isna(), 'day'] = test_df.loc[test_df['recur'].isna(), 'followup']
    test_df = test_df.drop(columns=['recur', 'followup'])
    test_df = test_df[test_df['day'] > 0]
    test_df['day'] = test_df['day'] // 180 + 1
    return train_df, val_df, test_df


def label_cluster(feature, cluster):
    clusters = defaultdict()
    cluster_method = type(cluster).__name__
    # need to

    i = 0
    for slide, tiles_in_feature_space in tqdm(feature.items()):
        # if i > 30:
        #     break
        i += 1
        if cluster_method == 'GaussianMixture':
            #clusters[slide] = cluster.predict([x[0] for x in tiles_in_feature_space])
            clusters[slide] = cluster.predict_proba([x[0] for x in tiles_in_feature_space])
        else:
            pass
            # clusters[c] = cluster.predict(feature[int(k)])
    return clusters


def load_data(data_dir, cluster_dir, normalize='mean', cls=1):
    split_dir = data_dir.rsplit('/', 1)[0] + '/'
    cluster = pickle.load(open(cluster_dir, 'rb'))
    cluster_method = type(cluster).__name__
    if cluster_method == 'GaussianMixture':
        n_clusters = len(cluster.weights_)
    else:
        n_clusters = cluster.n_clusters
    files = os.listdir(data_dir)
    train_features = pickle.load(
        open(os.path.join(data_dir, list(filter(lambda s: "train" in s and "embedding" in s, files))[0]), 'rb'))
    train_outcomes = pickle.load(
        open(os.path.join(data_dir, list(filter(lambda s: "train" in s and "outcomes" in s, files))[0]), 'rb'))
    val_features = pickle.load(
        open(os.path.join(data_dir, list(filter(lambda s: "val" in s and "embedding" in s, files))[0]), 'rb'))
    val_outcomes = pickle.load(
        open(os.path.join(data_dir, list(filter(lambda s: "val" in s and "outcomes" in s, files))[0]), 'rb'))
    test_features = pickle.load(
        open(os.path.join(data_dir, list(filter(lambda s: "test" in s and "embedding" in s, files))[0]), 'rb'))
    test_outcomes = pickle.load(
        open(os.path.join(data_dir, list(filter(lambda s: "test" in s and "outcomes" in s, files))[0]), 'rb'))

    train_tcga_flag = np.array([])
    val_tcga_flag = np.array([])
    test_tcga_flag = np.array([])

    train_cluster = label_cluster(train_features, cluster)
    val_cluster = label_cluster(val_features, cluster)
    test_cluster = label_cluster(test_features, cluster)

    train_data = transform(train_features, train_cluster, train_outcomes, n_clusters, normalize, demo=False)
    val_data = transform(val_features, val_cluster, val_outcomes, n_clusters, normalize, demo=False)
    test_data = transform(test_features, test_cluster, test_outcomes, n_clusters, normalize, demo=False)

    return train_data, val_data, test_data


def counter(arr, n):
    count = defaultdict(lambda: 0)
    for k, v in Counter(arr).items():
        count[k] = v
    return [count[i] for i in range(n)]


def transform(features, cluster, outcomes, n_clusters, normalize='count', cls=1, weight=None, demo=None):
    count_list = []
    outcome_list = []
    recur_day_list = []
    followup_day_list = []
    tcga_flag_list = []
    cluster_method = type(cluster).__name__
    for slide, cluster_ids in cluster.items():
        slide_id = "-".join(slide.split('-')[0:3])
        if normalize == 'mean':
            count_list.append(cluster_ids.mean(axis=0))
        else:
            pass
            #count_list.append(counter(cluster[int(cluster_id)], n_clusters))

        outcome_list.append(outcomes[slide_id]['recurrence'])
        recur_day_list.append(int(outcomes[slide_id]['recurrence_free_days']) if outcomes[slide_id]['recurrence_free_days'] else None)
        followup_day_list.append(int(outcomes[slide_id]['followup_days']) if (outcomes[slide_id]['followup_days'] and outcomes[slide_id]['followup_days'].isnumeric()) else None)
    count_list = np.array(count_list)
    outcome_list = np.array(outcome_list)
    count_list = count_list.astype(np.float)

    if normalize == 'mean':
        cluster_features = (count_list.T / count_list.sum(axis=1)).T
    elif normalize == 'count':
        cluster_features = (count_list.T / count_list.sum(axis=1)).T
    elif normalize == 'onehot':
        cluster_features = (count_list > 1e-5)
    elif normalize == 'avg':
        cluster_features = count_list
    elif normalize == 'weight':
        cluster_features = count_list * weight
    elif normalize == 'sum':
        cluster_features = count_list
    return {'cluster': cluster_features,
            'recur_day': np.array(recur_day_list),
            'followup_day': np.array(followup_day_list),
            'outcome': outcome_list,
            }
