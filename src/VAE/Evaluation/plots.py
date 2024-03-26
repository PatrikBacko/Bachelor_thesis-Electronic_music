import matplotlib.pyplot as plt
import plotly

import numpy as np
import sklearn

cm = 1/2.54

def plot_means_dist_hist(means_and_logvars_dict, output_path):

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(10*cm, 20*cm))

    dists = []
    for group in means_and_logvars_dict:
        group_dists = [np.linalg.norm(sample['mean'] - np.zeros((sample['mean'].shape))) for sample in means_and_logvars_dict[group]]
        ax[0].hist(group_dists, bins=300, alpha=1, label=group)

        dists.extend(group_dists)
        
    ax[0].set_title('Distance from zero vector for each sample groupwise')
    ax[0].xlabel('Distance')
    ax[0].ylabel('Count')
    ax[0].legend()

    ax[1].hist(dists, bins=300)
    ax[1].set_title('Distance from zero vector for all samples')
    ax[1].xlabel('Distance')
    ax[1].ylabel('Count')

    plt.savefig(output_path / 'means_dist_hist.png')


def plot_3D_reduced_dims(means_and_logvars_dict, output_path):
    pca = sklearn.decomposition.PCA(n_components=3)
    tsne = sklearn.manifold.TSNE(n_components=3)

    means = [mean for group in means_and_logvars_dict for mean in means_and_logvars_dict[group]]
    pca.fit(means)
    tsne.fit(means)
    

    trace = []

    for group in means_and_logvars_dict:
        means = [sample['mean'] for sample in means_and_logvars_dict[group]]
        pca_reduced_means = pca.transform(means)
        tsne_reduced_means = tsne.transform(means)
        
        trace.append(plotly.graph_objs.Scatter3d(
            x=pca_reduced_means[:, 0],
            y=pca_reduced_means[:, 1],
            z=pca_reduced_means[:, 2],
            mode='markers',
            marker=dict(
            size=5,
            color=group,
            colorscale='Viridis',
            opacity=0.8
            ),
            name=group
        ))