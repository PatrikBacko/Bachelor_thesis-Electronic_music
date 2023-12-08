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
import os

import keyboard

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

    #argument for noise
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
    

    #TODO: both spectogram and wave plotting in the future ? 
    parser.add_argument("-t", "--plot", help="plot the spectogram or waves, depending on settings (original, roundtripped and noised)", choices=["spectogram", "waves", "both"], default=False)
    parser.add_argument("-p", "--play", help="play the original, roundtripped and noised waves", action="store_true", default=False)

    #TODO: batch ??????
    parser.add_argument("-b", "--batch", help="process a batch of files", action="store_true", default=False)

    #TODO: save ???
    parser.add_argument("-s", "--save", help="save rountrip with or without noise (or both)", choices=["noised", "round_trip", "both"], default=False)


    return parser


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

def play_wave(wave: np.ndarray, sample_rate: int) -> None:
    """
    Play wave data using pyaudio

    params:
        - wave: wave data
        - sample_rate: wave sample rate
    """

    p = pyaudio.PyAudio()

    # open a stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    stream.write(wave.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()




#mfcc
def get_mfcc(wave: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """
    Extract mfcc features from wave data

    params:
        - wave: wave data
        - sample_rate: wave sample rate
    
    returns:
        - mfcc spectogram
    """
    return lb.feature.mfcc(y=wave, sr=sample_rate, **kwargs)


def mfcc_roundtrip(wave: np.ndarray, sample_rate: int, noise_function: callable, **kwargs) -> np.ndarray:
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
    mfcc = get_mfcc(wave, sample_rate, **kwargs)
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



def normal_noise(mean: float, variance: float, shape: tuple[int, int], dtype = np.float32) -> np.ndarray:
    """
    Generate normal noise with given mean and variance

    params:
        - mean: mean of the normal distribution
        - variance: variance of the normal distribution
        - shape: shape of the noise array
    
    returns:
        - noise array
    """
    return np.random.normal(mean, variance, shape, dtype=dtype)

def uniform_noise(mean: float, variance: float, shape: tuple[int, int], dtype=np.float32) -> np.ndarray:
    """
    Generate uniform noise with given mean and variance

    params:
        - mean: mean of the uniform distribution
        - variance: variance of the uniform distribution
        - shape: shape of the noise array

    returns:
        - noise array
    """
    return np.random.uniform(mean-variance, mean+variance, shape, dtype=dtype)

def constant_noise(mean: float, variance: float, shape: tuple[int, int], dtype = np.float32) -> np.ndarray:
    """
    Generate constant noise with given mean and variance

    params:
        - mean: mean of the constant distribution
        - variance: variance of the constant distribution
        - shape: shape of the noise array 

    returns:
        - noise array
    """
    return np.full(shape, mean, dtype=dtype)


def add_noise(spectogram, noise) -> callable: 
    """
    Add noise to the spectogram

    params:
        - spectogram: spectogram to add noise to
        - noise: noise array
    
    returns:
        - spectogram with added noise
    """
    return spectogram + noise

def multiply_noise(spectogram, noise) -> callable:
    """
    Multiply the spectogram with noise

    params:
        - spectogram: spectogram to multiply with noise
        - noise: noise array
    
    returns:
        - spectogram multiplied with noise
    """
    return spectogram * noise

def generate_noise(args) -> callable:
    """
    Generate noise function based on the given arguments

    params:
        - args: arguments for generating noise function

    returns:
        - noise function
    """

    #noise generating distribution
    if args.distribution == "normal":
        distribution = lambda shape: normal_noise(args.mean, args.variance, shape)
    elif args.distribution == "uniform":
        distribution = lambda shape: uniform_noise(args.mean, args.variance, shape)
    elif args.distribution == "constant":
        distribution = lambda shape: constant_noise(args.mean, args.variance, shape)

    #noise scope
    if args.scope == "pixel":
        scope = lambda spectogram: distribution(spectogram.shape)
    elif args.scope == "column":
        scope = lambda spectogram: np.outer(distribution(spectogram.shape[1]), np.zeros(spectogram.shape[0]))
    elif args.scope == "row":
        scope = lambda spectogram: np.outer(np.zeros(spectogram.shape[1]), distribution(spectogram.shape[0]))
    elif args.scope == "entire_picture":
        scope = lambda spectogram: np.full(spectogram.shape, distribution((1,))[0])

    #noise operation type
    if args.operation == "additive":
        noise_function = lambda spectogram: add_noise(spectogram, scope(spectogram))
    elif args.operation == "multiplicative":
        noise_function = lambda spectogram: multiply_noise(spectogram, scope(spectogram))

    return noise_function



def plot_waves(wave: np.ndarray, round_trip: np.ndarray, round_trip_noised: np.ndarray,  sample_rate: int) -> None:

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    librosa.display.waveplot(wave, sr=sample_rate)
    plt.title("Original Wave")

    plt.subplot(1, 3, 2)
    librosa.display.waveplot(round_trip, sr=sample_rate)
    plt.title("Round Trip Wave")

    plt.subplot(1, 3, 3)
    librosa.display.waveplot(round_trip_noised, sr=sample_rate)
    plt.title("Noised Round Trip Wave")

    plt.show()

def plot_spectograms(spectogram_1, spectogram_2) -> None:
    #TODO: implement
    raise NotImplementedError("Spectogram plotting is not implemented yet")


def main(args):
    #load wave file
    wave, sample_rate = load_lb_wave(args.input)

    no_noise = lambda x: x

    #TODO: add noise functions
    if args.noise:
        noise_function = generate_noise(args)
    else:
        noise_function = no_noise


    if args.feature == "mfcc":
        round_trip_noised = mfcc_roundtrip(wave, sample_rate, noise_function)
        round_trip = mfcc_roundtrip(wave, sample_rate, no_noise)

    elif args.feature == "mel":
        round_trip_noised = mel_spectogram_roundtrip(wave, sample_rate, noise_function)
        round_trip = mel_spectogram_roundtrip(wave, sample_rate, no_noise)

    elif args.feature == "stft":
        round_trip_noised = stft_roundtrip(wave, sample_rate, noise_function)
        round_trip = stft_roundtrip(wave, sample_rate, no_noise)


    if args.plot == "spectogram":
        plot_spectograms(round_trip, round_trip_noised)
    elif args.plot == "waves":
        plot_waves(wave, round_trip, round_trip_noised, sample_rate)
    elif args.plot == "both":
        raise NotImplementedError("Both plotting is not implemented yet") 

    if args.play:
        #play waves
        play_wave(wave, sample_rate)
        play_wave(round_trip, sample_rate)
        play_wave(round_trip_noised, sample_rate)
    
    #TODO: save round trip wave file
    # if args.save:
    #     if args.save
    #     save_lb_wave(round_trip, sample_rate, args.output)

    if args.save == "noised":
        save_lb_wave(round_trip_noised, sample_rate, args.output +"\\" +os.path.basename(args.input) + "_noised.wav")
    elif args.save == "round_trip":
        pass


if __name__ == "__main__":
    args = build_arguments().parse_args()

    main(args)