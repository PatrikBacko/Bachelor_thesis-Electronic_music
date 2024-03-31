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
    eval_dir_path = model_dir_path / 'evaluation'


    config = load_config(model_dir_path / 'config.json')
    model = load_model(model_dir_path / 'model.pkl', config.model, config.latent_dim)

    # Job 1
    reconstruct_samples(model, config, eval_dir_path / 'reconstructed_samples', data_path=args.data_path, n_samples=10)

    # Job 2
    generate_and_save_means_and_logvars(model, config, eval_dir_path, args.data_path)

    # Job 3
    means_logvars_dict = load_means_logvars_json(eval_dir_path)
    make_plots(means_logvars_dict, eval_dir_path)







if __name__ == '__main__':
    main()