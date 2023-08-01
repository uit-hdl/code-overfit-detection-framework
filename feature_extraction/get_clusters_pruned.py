hausdorf_l = np.zeros((len(train_features.keys()), len(train_features.keys())))
np.fill_diagonal(hausdorf_l, np.nan)
frechet_l = np.zeros((len(train_features.keys()), len(train_features.keys())))
np.fill_diagonal(frechet_l, np.nan)
# from Data Representativity for Machine Learning and AI Systems
wasserstein_l = np.zeros((len(train_features.keys()), len(train_features.keys())))
np.fill_diagonal(wasserstein_l, np.nan)
# TODO: shannon coverage? look at all other values and measure coverage of those
for ((i,left),(j, right)) in combinations(enumerate(slide_sets), 2):
    haus_d = max(directed_hausdorff(left, right)[0], directed_hausdorff(right, left)[0])
    hausdorf_l[i][j] = haus_d
    hausdorf_l[j][i] = haus_d


    # TODO: frechet
    l_s = left.shape[0]
    r_s = right.shape[0]

    # Extend the smallest of left or right with the contents from the bigger one (giving distance=0)
    # FIXME: this is equivalent to dropping of points................
    if l_s < r_s:
        # tail = right[-(r_s-l_s):]
        # left = np.append(left, tail, axis=0)
        right = right[-l_s:]
    else:
        # tail = left[-(l_s-r_s):]
        # right = np.append(right, tail, axis=0)
        left = left[-r_s:]

    frdist = frechetdist.frdist(left, right)
    frechet_l[i][j] = frdist
    frechet_l[j][i] = frdist

    #https://stackoverflow.com/a/57563383
    cd = cdist(left, right)
    assignment = linear_sum_assignment(cd)
    wd = cd[assignment].sum() / left.shape[0]
    wasserstein_l[i][j] = wd
    wasserstein_l[j][i] = wd


labels = list(map(lambda l: 'Slide %d' % (l+1), range(hausdorf_l.shape[0]))) + ['mean']
mean_hausdorf = np.nanmean(hausdorf_l)
hausdorf_l = np.vstack([hausdorf_l, np.nanmean(hausdorf_l, axis=1)])
hausdorf_l = np.pad(hausdorf_l, ((0,0),(0,1)), mode='constant', constant_values=mean_hausdorf)
hdf = pd.DataFrame(hausdorf_l, columns=labels).to_csv(os.path.join(args.out_dir, 'hausdorf.csv'))
mean_frechet = np.nanmean(frechet_l)
frechet_l = np.vstack([frechet_l, np.nanmean(frechet_l, axis=1)])
frechet_l = np.pad(frechet_l, ((0,0),(0,1)), mode='constant', constant_values=mean_frechet)
hdf = pd.DataFrame(frechet_l, columns=labels).to_csv(os.path.join(args.out_dir, 'frechet.csv'))
mean_wd = np.nanmean(wasserstein_l)
wasserstein_l = np.vstack([wasserstein_l, np.nanmean(wasserstein_l, axis=1)])
wasserstein_l = np.pad(wasserstein_l, ((0,0),(0,1)), mode='constant', constant_values=mean_wd)
hdf = pd.DataFrame(wasserstein_l, columns=labels).to_csv(os.path.join(args.out_dir, 'wasserstein.csv'))

metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
# metrics_l = np.zeros((len(train_features.keys())), len(metrics))
metrics_l = np.zeros((len(train_features.keys()), len(metrics)))
for ((i,points), (j,metric)) in product(enumerate(slide_sets), enumerate(metrics)):
    metrics_l[i][j] = np.mean(pdist(points, metric))

# TODO: extract mean
df = pd.DataFrame(metrics_l, columns=metrics)
df.to_csv(os.path.join(args.out_dir, 'out.csv'), sep='\t', encoding='utf-8')
print(df)
# plt.show()

