import librosa as lb
import numpy as np
import sys

from src.VAE.utils.converters.converter import Converter
from src.VAE.exceptions.InvalidSamplingException import InvalidInverseConversionException


class MfccConverter(Converter):
    
    MFCC_KWARGS = {
    'n_mfcc': 512,
    'dct_type': 2,
    'norm': "ortho",
    'lifter': 0,

    #mel spectrogram kwargs
    'n_fft': 512,  
    'hop_length': 256,
    'win_length': 512,
    'window': "hann",
    'center': True,
    'pad_mode': "constant",
    'power': 2.0,

    #mel filterbank kwargs
    'n_mels': 256,
    'fmin': 0.0,
    'fmax': None,
    'htk': False
    }

    def get_default_config(self):
        '''
        Gets the default config for the mfcc conversion

        returns:
            dict - default config for the mfcc conversion
        '''
        config = {
            'type': 'mfcc',
            'kwargs': MfccConverter.MFCC_KWARGS,
            'channels': 1,
            'height': MfccConverter.MFCC_KWARGS['n_mels'],
        }

        return config
    
    def convert_wave_to_spectogram(self, wave, sr, spectogram_kwargs = None):
        '''
        Gets a wave and its sample rate, then returns its mfcc

        params:
            wave (np.array) - wave to convert to mfcc
            sr (int) - sample rate of the wave
            spectogram_kwargs (dict) - kwargs for the mfcc conversion

        returns:
            np.ndarray - mfcc of the wave
        '''
        if not spectogram_kwargs: spectogram_kwargs = MfccConverter.MFCC_KWARGS

        return lb.feature.mfcc(y=wave, sr=sr, **spectogram_kwargs)
    

    def convert_spectogram_to_wave(self, spectogram, sr, spectogram_kwargs = None):
        '''
        Gets a mfcc and returns its inverse

        params:
            mfcc (np.ndarray) - mfcc to get inverse of
            sr (int) - sample rate of the wave
            spectogram_kwargs (dict) - kwargs for the spectogram conversion

        returns:
            np.ndarray - inverse of the mfcc

        raises:
            InvalidSamplingException - if there is an error with mfcc inverse conversion
        '''
        if not spectogram_kwargs: spectogram_kwargs = MfccConverter.MFCC_KWARGS

        inverse_mfcc_kwargs = self._get_inverse_mfcc_kwargs(spectogram_kwargs)

        try:
            return lb.feature.inverse.mfcc_to_audio(mfcc = spectogram, sr = sr, **inverse_mfcc_kwargs)

        except Exception as e:
            print(e, file=sys.stderr)
            raise InvalidInverseConversionException('Error with mfcc conversion')
        

    def pad_or_trim_spectogram(self, spectogram, length):
        '''
        Pads or trims the spectogram to the desired length

        params:
            spectogram (np.ndarray) - spectogram to pad or trim
            length (int) - length to pad or trim the spectogram

        returns:
            np.ndarray - padded or trimmed spectogram
        '''

        if spectogram.shape[1] > length:
            spectogram =  spectogram[:, :length]
        else:
            last_column = spectogram[:, -1:]
            padding = np.repeat(last_column, length - spectogram.shape[1], axis=1)

            spectogram = np.concatenate((spectogram, padding), axis=1)

        return spectogram
        
        
    def _get_inverse_mfcc_kwargs(self, mfcc_kwargs = None):
        '''
        Gets mfcc kwargs and returns kwargs for inverse mfcc conversion

        params:
            inverse_mfcc_kwargs (dict) - kwargs for the inverse mfcc

        returns:
            dict - inverse mfcc kwargs
        '''

        if not mfcc_kwargs: mfcc_kwargs = MfccConverter.MFCC_KWARGS

        return {
            'n_mels': mfcc_kwargs['n_mels'],
            'dct_type': mfcc_kwargs['dct_type'],
            'norm': mfcc_kwargs['norm'],
            'lifter': mfcc_kwargs['lifter'],
            'n_fft': mfcc_kwargs['n_fft'],
            'hop_length': mfcc_kwargs['hop_length'],
            'win_length': mfcc_kwargs['win_length'],
            'window':  mfcc_kwargs['window'],
            'center': mfcc_kwargs['center'],
            'pad_mode': mfcc_kwargs['pad_mode'],
            'power': mfcc_kwargs['power'],


            'ref': 1.0,
            'n_iter': 32,
            'length': None,
            'dtype': np.float32
        }