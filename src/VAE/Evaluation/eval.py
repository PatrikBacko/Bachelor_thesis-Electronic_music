import os

import numpy as np
import librosa as lb
# import soundfile as sf
# import pyaudio
import sounddevice as sd

import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn
import sklearn.decomposition

import matplotlib.pyplot as plt

from models.VAE_1 import VAE_1






from utils.config import load_config
from models.load_model import load_model
from reconstruct_samples import reconstruct_samples
from generate_means_logvars import generate_and_save_means_and_logvars


from pathlib import Path
from typing import Sequence

import argparse

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












if __name__ == '__main__':
    main()