#!/usr/bin/env python3

"""
This script is used to create a round trip (wav -> extract features -> (optionally) add noise -> wav) and save the results to a .wav file. (or a batch of files when given a directory)
The scripts tries several feature extraction methods:
- MFCC
- Mel Spectogram
- STFT

Types of noise:
- Generating distribution:
    -- normal 
    -- uniform
- operation type:
    -- additive
    -- multiplicative (coefficients)
- scope of noise:
    -- pixel 
    -- time step (column)
    -- frequency step (row)


"""


import librosa as lb
import librosa.display
import librosa.feature
import librosa.util
import librosa.effects

import numpy as np

import matplotlib.pyplot as plt

import wave 
import soundfile as sf
import pyaudio

import argparse

FEATURE_EXTRACTION_METHODS = ["mfcc", "mel", "stft"]

NOISE_GENERATING_DISTS = ["normal", "uniform", "constant"]
NOISE_OPERATION_TYPES = ["additive", "multiplicative"]
NOISE_SCOPE = ["pixel", "column", "row", "entire_picture"]


def build_arguments() -> argparse.ArgumentParser:
    """
    Build the arguments for the script
    """
    parser = argparse.ArgumentParser(description=__doc__)
    #argument for input path to wav file or directory with wav files
    parser.add_argument("input", help="input path to wav file")
    #argument for output path to wav file or directory with wav files
    parser.add_argument("output", help="output path to a directory for output wav files", default="./output")


    parser.add_argument("-n", "--noise", help="add noise to the spectogram", action="store_true", default=False)

    #argument for noise type
    parser.add_argument("-d", "--distribution", help="noise generating distribution"
                        "normal: normal distribution"
                        "uniform: uniform distribution"
                        "constant: value is equal to the chosen mean"
                        , choices=NOISE_GENERATING_DISTS, default="normal")
    #argument for noise operation type
    parser.add_argument("-o", "--operation", help="noise operation type (how will be noise added to the spectogram)"
                        "additive: add noise to the spectogram"
                        "multiplicative: multiply the spectogram with noise (noise values are coefficients)"
                        , choices=NOISE_OPERATION_TYPES, default="additive")
    #argument for noise scope
    parser.add_argument("-s", "--scope", help="noise scope."
                        "pixel: add noise to each pixel in the spectogram"
                        "column: each column in the spectogram will have the same noise, but different from other columns"
                        "row: each row in the spectogram will have the same noise, but different from other rows"
                        "entire_picture: the entire spectogram will have the same noise"
                        , choices=NOISE_SCOPE, default="pixel")
    
    #argument for noise variance
    parser.add_argument("-v", "--variance", help="noise variance for generating distribution", type=float, default=0.1)

    #argument for noise mean
    parser.add_argument("-m", "--mean", help="noise mean for generating distribution", type=float, default=0.0)

    #aguemnt for feature extraction method
    parser.add_argument("-f", "--feature", help="feature extraction method"
                        "mfcc: Mel-frequency cepstral coefficients"
                        "mel: Mel Spectogram"
                        "stft: Short-time Fourier transform"
                        , choices=FEATURE_EXTRACTION_METHODS, default="mfcc")


# * kwargs

# mfcc parameters
S=None
n_mfcc=20
dct_type=2
norm='ortho'
lifter=0

# mel spectogram parameters
S=None
n_fft=2048, 
hop_length=512
win_length=None
window='hann'
center=True
pad_mode='constant'
power=2.0

#mel parameters
n_mels=128 
fmin=0.0 
fmax=None
htk=False
norm='slaney'
dtype= np.float32

# # stft parameters
# n_fft=2048
# hop_length=None
# win_length=None
# window='hann'
# center=True
# dtype=None
# pad_mode='constant'
# out=None



#utils
def load_lb_wave(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load wave file using librosa

    params:
        - file_path: path to wave file
    
    returns:
        - tuple of wave data and sample rate
    """
    return lb.load(file_path)

def get_wav_info(wav_file_path: str) -> sf._SoundFileInfo:
    """
    get wave file info using soundfile

    params:
        - wav_file_path: path to wave file
    
    returns:
        - Soundfile wave file info
    """
    return sf.info(wav_file_path)

def save_lb_wave(wave: np.ndarray, sample_rate: int, output_file_path: str) -> None:
    """
    Save wave data to a wave file using librosa

    params:
        - wave: wave data
        - sample_rate: wave sample rate
        - output_file_path: path to save the wave file
    """
    lb.output.write_wav(output_file_path, wave, sample_rate)





#mfcc
def get_mfcc(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract mfcc features from wave data

    params:
        - wave: wave data
        - sample_rate: wave sample rate
    
    returns:
        - mfcc spectogram
    """
    return lb.feature.mfcc(y=wave, sr=sample_rate)


def mfcc_roundtrip(wave: np.ndarray, sample_rate: int, noise_function: callable) -> np.ndarray:
    """
    Create a round trip (wav -> extract features -> add noise -> wav) using mfcc feature extraction method

    params:
        - wave: wave data
        - sample_rate: wave sample rate
        - noise: noise function
    
    returns:
        - round trip wave data
    """
    #extract mfcc features
    mfcc = get_mfcc(wave, sample_rate)
    #add noise to the spectogram
    mfcc_noised = noise_function(mfcc)
    #invert the mfcc spectogram
    return lb.feature.inverse.mfcc_to_audio(mfcc_noised, sr = sample_rate)



#mel spectogram
def get_mel_spectogram(wave: np.ndarray, sample_rate: int) -> np.ndarray:  
    """
    Extract mel spectogram from wave data

    params:
        - wave: wave data
        - sample_rate: wave sample rate
    
    returns:
        - mel spectogram
    """
    return lb.feature.melspectrogram(y=wave, sr=sample_rate)

def mel_spectogram_roundtrip(wave: np.ndarray, sample_rate: int, noise_function: callable) -> np.ndarray:
    """
    Create a round trip (wav -> extract features -> add noise -> wav) using mel spectogram feature extraction method

    params:
        - wave: wave data
        - sample_rate: wave sample rate
        - noise: noise function
    
    returns:
        - round trip wave data
    """
    #extract mel spectogram
    mel_spectogram = get_mel_spectogram(wave, sample_rate)
    #add noise to the spectogram
    mel_spectogram_noised = noise_function(mel_spectogram)
    #invert the mel spectogram
    return lb.feature.inverse.mel_to_audio(mel_spectogram_noised, sr = sample_rate)



#stft
def get_stft(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract stft spectogram from wave data

    params:
        - wave: wave data
        - sample_rate: wave sample rate
    
    returns:
        - stft spectogram
    """
    return lb.stft(y=wave, sr=sample_rate)

def split_stft(stft_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split stft spectogram into magnitude and phase. to join them back use np.multiply(magnitude, phase)

    params:
        - stft_features: stft spectogram

    returns:
        - tuple of magnitude and phase
    """
    magnitude, phase = lb.magphase(stft_features)
    return magnitude, phase

def stft_roundtrip(wave: np.ndarray, sample_rate: int, noise_function: callable) -> np.ndarray:
    """
    Create a round trip (wav -> extract features -> add noise -> wav) using stft spectogram feature extraction method

    params:
        - wave: wave data
        - sample_rate: wave sample rate
        - noise: noise function
    
    returns:
        - round trip wave data
    """
    #extract stft spectogram
    stft = get_stft(wave, sample_rate)

    #split stft spectogram into magnitude and phase
    stft_magnitude, stft_phase = split_stft(stft)

    #add noise to the spectogram
    stft_magnitude_noised = noise_function(stft_magnitude)

    stft_noised = np.multiply(stft_magnitude_noised, stft_phase)

    #invert the stft spectogram
    return lb.istft(stft_noised, sr = sample_rate)


def main(args):
    #load wave file
    wave, sample_rate = load_lb_wave(args.input)

    noise_function = lambda x: x

    if args.feature == "mfcc":
        round_trip = mfcc_roundtrip(wave, sample_rate, noise_function)
    elif args.feature == "mel":
        round_trip = mel_spectogram_roundtrip(wave, sample_rate, noise_function)
    elif args.feature == "stft":
        round_trip = stft_roundtrip(wave, sample_rate, noise_function)

    #save round trip wave file
    save_lb_wave(round_trip, sample_rate, args.output)



if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # args = parser.parse_args()