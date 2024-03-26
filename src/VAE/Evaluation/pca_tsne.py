import sklearn


def get_fitted_pca(means_logvars_dict, n_components):
    pca = sklearn.decomposition.PCA(n_components=n_components)
    means = [mean for group in means_logvars_dict for mean in means_logvars_dict[group]]
    pca.fit(means)
    return pca

def get_fitted_tsne(means_logvars_dict, n_components):
    tsne = sklearn.manifold.TSNE(n_components=n_components)
    means = [mean for group in means_logvars_dict for mean in means_logvars_dict[group]]
    tsne.fit(means)
    return tsne