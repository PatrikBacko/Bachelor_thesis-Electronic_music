import matplotlib
import matplotlib.pyplot as plt
import plotly

import sklearn.manifold

from pathlib import Path
import numpy as np


from src.VAE.evaluation.pca import get_fitted_pca

cm = 1/2.54

def plot_means_dist_hist(means_and_logvars_dict, output_path):
    '''
    plots histogram of euclidean distances of means from zero vector. saves the plot as png file to the output_path.

    params:
        means_and_logvars_dict: (dict), dictionary of means and logvars of samples
        output_path: (Path), path to the directory where to save the plot

    returns:
        None
    '''

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(40*cm, 20*cm))

    dists = []
    for group in means_and_logvars_dict:
        group_dists = [np.linalg.norm(sample['mean'] - np.zeros((sample['mean'].shape))) for sample in means_and_logvars_dict[group]]
        ax[0].hist(group_dists, bins=100, alpha=1, label=group)

        dists.extend(group_dists)
        
    ax[0].set_title('Distance from zero vector for each sample groupwise')
    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Count')
    ax[0].legend()

    ax[1].hist(dists, bins=100)
    ax[1].set_title('Distance from zero vector for all samples')
    ax[1].set_xlabel('Distance')
    ax[1].set_ylabel('Count')

    plt.savefig(output_path / 'means_dist_hist.png')


def plot_3D_reduced_dims_tsne(means_logvars_dict, output_path):
    '''
    plots 3D scatter plot of reduced dimensions of means using TSNE saves the plot as html files to the output_path.

    params:
        means_logvars_dict: (dict), dictionary of means and logvars of samples
        output_path: (Path), path to the directory where to save the plots

    returns:
        None
    '''

    means_group = [(sample['mean'], group) for group in means_logvars_dict for sample in means_logvars_dict[group]]
    transf_means = sklearn.manifold.TSNE(n_components=3).fit_transform(np.array([mean for mean, _ in means_group]))
    transf_means_group = list(zip(transf_means, [group for _, group in means_group]))
    transf_means_dict = {group: [mean for mean, g in transf_means_group if g == group] for group in means_logvars_dict.keys()}

    trace_tsne = []
    colors = [matplotlib.colors.rgb2hex(color) for color in plt.cm.tab10.colors]

    for i, group in enumerate(transf_means_dict.keys()):
        means = transf_means_dict[group]
        
        trace_tsne.append(plotly.graph_objs.Scatter3d(
            x=[mean[0] for mean in means],
            y=[mean[1] for mean in means],
            z=[mean[2] for mean in means],
            mode='markers',
            marker=dict(
            size=5,
            color=colors[i],
            opacity=0.8
            ),
            name=group
        ))
    
    fig_tsne = plotly.graph_objs.Figure(data=trace_tsne)
    
    fig_tsne.update_layout(title='Tsne reduced dimensions')

    fig_tsne.write_html(str(output_path / 'tsne_reduced_dims.html'))


def plot_3D_reduced_dims_pca(means_logvars_dict, output_path):
    '''
    plots 3D scatter plot of reduced dimensions of means using PCA saves the plot as html files to the output_path.

    params:
        means_logvars_dict: (dict), dictionary of means and logvars of samples
        output_path: (Path), path to the directory where to save the plots

    returns:
        None
    '''

    pca = get_fitted_pca(means_logvars_dict, 3)

    trace_pca = []
    colors = [matplotlib.colors.rgb2hex(color) for color in plt.cm.tab10.colors]

    for i, group in enumerate(means_logvars_dict.keys()):
        means = [sample['mean'] for sample in means_logvars_dict[group]]
        pca_reduced_means = pca.transform(means)
        
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

    fig_pca = plotly.graph_objs.Figure(data=trace_pca)

    fig_pca.update_layout(title='Pca reduced dimensions')

    fig_pca.write_html(str(output_path / 'pca_reduced_dims.html'))


def plot_pca_variance_ratio(means_logvars_dict, output_path):
    '''
    Plots explained variance ratio for each principal component and saves the plot as png file to the output_path.

    params:
        means_logvars_dict: (dict), dictionary of means and logvars of samples
        output_path: (Path), path to the directory where to save the plot

    returns:
        None
    '''


    pca = get_fitted_pca(means_logvars_dict, None)

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20*cm, 20*cm))
    
    ax.plot(pca.explained_variance_ratio_)
    ax.set_title('Explained variance ratio for each principal component')
    ax.set_xlabel('Principal component')
    
    ax.set_ylabel('Explained variance ratio')

    plt.savefig(output_path / 'pca_variance_ratio.png')



def make_plots(means_logvars_dict, output_path):
    '''
    makes all the plots for the evaluation of the model and saves them to the output_path.

    plots:
        - histogram of euclidean distances of means from zero vector
        - 3D scatter plot of reduced dimensions of means using PCA and TSNE
        - explained variance ratio for each principal component

    params:
        means_logvars_dict: (dict), dictionary of means and logvars of samples
        output_path: (Path), path to the directory where to save the plots

    returns:
        None
    '''

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)


    plot_means_dist_hist(means_logvars_dict, output_path)
    plot_3D_reduced_dims_pca(means_logvars_dict, output_path)
    plot_3D_reduced_dims_tsne(means_logvars_dict, output_path)
    plot_pca_variance_ratio(means_logvars_dict, output_path)