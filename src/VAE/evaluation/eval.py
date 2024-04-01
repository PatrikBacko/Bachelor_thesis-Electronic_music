'''
Script for evaluating a trained VAE model.

Evaluation jobs:
1. Reconstruct samples from the data. (test data, and randomly sampled from train-set)

2. Sample convex combinations of random samples in latent space and reconstruct them.

3. Generate and save means and logvars of samples from the data.

4. Make plots:
    - PCA reduced dimensions of means, 3D scatter plot
    - t-SNE reduced dimensions of means, 3D scatter plot
    - PCA explained variance ratio, plot
    - Histogram of euclidean distances of means from zero vector, plot

Usage:
    python eval.py <model_dir_path> <data_path>
'''

import sys
sys.path.append(r'C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music')

from pathlib import Path
from typing import Sequence
import os

import argparse


from src.VAE.utils.config import load_config
from src.VAE.models.load_model import load_model
from src.VAE.evaluation.reconstruct_samples import reconstruct_samples
from src.VAE.evaluation.generate_means_logvars import generate_and_save_means_and_logvars, load_means_logvars_json
from src.VAE.evaluation.plots import make_plots


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('model_dir_path', type=str, help='Path to directory of the model to evaluate.')
    parser.add_argument('data_path', type=str, help='Path to the data to evaluate on.')

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = parse_arguments()
    args = parser.parse_args(argv)

    model_dir_path = Path(args.model_dir_path)

    config = load_config(model_dir_path / 'config.json')
    model = load_model(model_dir_path / 'model.pkl', config.model, config.latent_dim)


    eval_dir_path = model_dir_path / 'evaluation'
    eval_dir_path.mkdir(exist_ok=True)

    # Job 1
    reconstruct_samples(model, config, eval_dir_path / 'reconstructed_samples', data_path=args.data_path, n_samples=10)

    # Job 2
    reconstruct_convex_combinations()


    # Job 3
    generate_and_save_means_and_logvars(model, config, eval_dir_path, args.data_path)

    # Job 4
    means_logvars_dict = load_means_logvars_json(eval_dir_path / 'means_logvars.json')
    make_plots(means_logvars_dict, eval_dir_path / 'plots')









if __name__ == '__main__':
    main()