#!/usr/bin/env python

import numpy as np
from scipy.stats import ttest_rel

seed = 1234
np.random.seed(seed)

# for t-tests: we investigate the results of the linear probe and ask whether or not the results are significantly different
LP_RESULTS_INCEPTION_OLD = [0.7, 0.65, 0.8, 0.75, 0.85, 0.69, 0.72]
LP_RESULTS_INCEPTION = [0.7, 0.65, 0.8, 0.75, 0.85, 0.69, 0.72]

LP_RESULTS_PHIKON = [-1, -1, -1, -1, -1, -1, -1]

t_test = ttest_rel(LP_RESULTS_INCEPTION, LP_RESULTS_PHIKON)
print(t_test)