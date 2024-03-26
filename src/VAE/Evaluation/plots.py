import matplotlib
import matplotlib.pyplot as plt
import plotly

import numpy as np
import sklearn


from src.VAE.evaluation.pca_tsne import get_fitted_pca, get_fitted_tsne
from src.VAE.evaluation.generate_means_logvars import load_means_logvars_dict

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


def plot_3D_reduced_dims(means_logvars_dict, pca, output_path):
    pca = get_fitted_pca(means_logvars_dict, 3)
    tsne = get_fitted_tsne(means_logvars_dict, 3)

    trace_pca = []
    trace_tsne = []
    colors = [matplotlib.colors.rgb2hex(color) for color in plt.cm.tab10.colors]

    for i, group in enumerate(means_logvars_dict.keys()):
        means = [sample['mean'] for sample in means_logvars_dict[group]]
        pca_reduced_means = pca.transform(means)
        tsne_reduced_means = tsne.transform(means)
        
        trace_pca.append(plotly.graph_objs.Scatter3d(
            x=pca_reduced_means[:, 0],
            y=pca_reduced_means[:, 1],
            z=pca_reduced_means[:, 2],
            mode='markers',
            marker=dict(
            size=5,
            color=colors[i],
            opacity=0.8
            ),
            name=group
        ))

        trace_tsne.append(plotly.graph_objs.Scatter3d(
            x=tsne_reduced_means[:, 0],
            y=tsne_reduced_means[:, 1],
            z=tsne_reduced_means[:, 2],
            mode='markers',
            marker=dict(
            size=5,
            color=colors[i],
            opacity=0.8
            ),
            name=group
        ))

    fig_pca = plotly.graph_objs.Figure(data=trace_pca)
    fig_tsne = plotly.graph_objs.Figure(data=trace_tsne)

    fig_pca.update_layout(
    title='Pca reduced dimensions')

    fig_tsne.update_layout(
    title='Tsne reduced dimensions')

    plotly.offline.plot(fig_pca, filename=output_path / 'pca_reduced_dims.html')
    plotly.offline.plot(fig_tsne, filename=output_path / 'tsne_reduced_dims.html')


def plot_pca_variance_ratio(means_logvars_dict, output_path):
    pca = get_fitted_pca(means_logvars_dict, None)
    
    plt.plot(pca.explained_variance_ratio_)
    plt.title('Explained variance ratio for each principal component')
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')

    plt.savefig(output_path / 'pca_variance_ratio.png')