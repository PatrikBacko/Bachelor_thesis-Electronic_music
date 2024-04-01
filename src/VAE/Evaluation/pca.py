import sklearn
import sklearn.decomposition

import numpy as np


def get_fitted_pca(means_logvars_dict, n_components):
    pca = sklearn.decomposition.PCA(n_components=n_components)
    means = np.array([sample['mean'] for group in means_logvars_dict for sample in means_logvars_dict[group]])
    pca.fit(means)
    return pca
