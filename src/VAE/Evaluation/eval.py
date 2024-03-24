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
from utils.prepare_data import pad_or_trim






from utils.config import load_config
from models.load_model import load_model



from pathlib import Path
from typing import Sequence

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('model_dir_path', type=str, help='Path to directory of the model to evaluate.')

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = parse_arguments()
    args = parser.parse_args(argv)

    model_dir_path = Path(args.model_dir_path)


    config = load_config(model_dir_path / 'config.json')
    model = load_model(model_dir_path / 'model.pkl', config.model, config.latent_dim)








if __name__ == '__main__':
    main()