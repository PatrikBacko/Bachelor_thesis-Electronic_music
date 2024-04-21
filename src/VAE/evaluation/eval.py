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
# sys.path.append(r'C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music')

from pathlib import Path
from typing import Sequence
import datetime

import argparse


from src.VAE.utils.config import load_config
from src.VAE.models.load_model import load_model

from src.VAE.evaluation.jobs.reconstruct_samples import reconstruct_samples
from src.VAE.evaluation.jobs.generate_means_logvars import generate_and_save_means_and_logvars, load_means_logvars_json
from src.VAE.evaluation.jobs.plots import make_plots
from src.VAE.evaluation.jobs.generate_convex_combinations import generate_convex_combinations
from src.VAE.evaluation.jobs.pca_shift import generate_samples_with_pca_shift
from src.VAE.evaluation.jobs.sample_random_waves import sample_and_save_random_waves


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('model_dir_path', type=str, help='Path to directory of the model to evaluate.')
    parser.add_argument('data_path', type=str, help='Path to the data to evaluate on.')

    parser.add_argument('-l', '--log_file', type=int, default=None, help='Path to a log file. If not given, logs will be printed to stdout.')

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = parse_arguments()
    args = parser.parse_args(argv)


    if args.log_file is not None:
        sys.stdout = open(args.log_file, 'w')

    start  = datetime.datetime.now()
    print(f'Date and time of training: {start.strftime("%Y-%m-%d %H:%M:%S")}')


    model_dir_path = Path(args.model_dir_path)

    config = load_config(model_dir_path / 'config.json')
    model = load_model(model_dir_path / 'model.pkl', config.model, config.latent_dim)
    model.eval()
    print(f'Loading model {config.model_name} in {model_dir_path}...')


    eval_dir_path = model_dir_path / 'evaluation'
    eval_dir_path.mkdir(exist_ok=True)


    jobs = {
        'reconstruct_samples': 
            lambda : reconstruct_samples(model, 
                                         config, 
                                         eval_dir_path / 'samples', 
                                         data_path=args.data_path, 
                                         n_samples=5),

        'generate_convex_combinations': 
            lambda : generate_convex_combinations(model, 
                                                  config, 
                                                  args.data_path, 
                                                  eval_dir_path / 'samples' / 'convex_combinations', 
                                                  test_samples=True, 
                                                  seed=42),

        'sample_random_waves': 
            lambda : sample_and_save_random_waves(model, 
                                                  config, 
                                                  eval_dir_path / 'samples' / 'sampled_random', 
                                                  n_samples=5, 
                                                  seed=None, 
                                                  sr=44_100),

        'generate_means_logvars': 
            lambda : generate_and_save_means_and_logvars(model, 
                                                         config, 
                                                         eval_dir_path, 
                                                         args.data_path),

        'make_plots': 
            lambda : make_plots(load_means_logvars_json(eval_dir_path / 'means_logvars.json'), 
                                eval_dir_path / 'plots'),

        'generate_pca_shift':
            lambda : generate_samples_with_pca_shift(model, 
                                                     config, 
                                                     load_means_logvars_json(eval_dir_path / 'means_logvars.json'), 
                                                     args.data_path, 
                                                     eval_dir_path / 'samples' / 'pca_shift', 
                                                     test_samples=True, 
                                                     seed=42)
    }


    for i, (job_name, job) in enumerate(jobs.items()):
        print(f'Job {i+1}: {job_name}')
        start_job = datetime.datetime.now()
        job()
        end_job = datetime.datetime.now()
        print(f'\tfinished in: {(end_job-start_job)}')
        print()


    print()
    end = datetime.datetime.now()
    print(f'Evaluation finished in: {(end-start)}')



if __name__ == '__main__':
    main()