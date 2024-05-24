import librosa as lb
import numpy as np


from src.VAE.utils.converters.converter import Converter


class StftConverter_gl(Converter):
    STFT_KWARGS = {
     'n_fft':512,
     'hop_length':256,
     'win_length':None,
     'window':'hann',
     'center':True,
     'dtype':None,
     'pad_mode':'constant',

    }

    
    def get_default_config(self, kwargs = None):
        '''
        Gets the default config for the stft conversion

        returns:
            dict - default config for the stft conversion
        '''
        if not kwargs: kwargs = StftConverter_gl.STFT_KWARGS

        return {
            'type': 'stft_gl',
            'kwargs': kwargs,
            'channels': 1,
            'height': kwargs['n_fft'] // 2 + 1
        }

    
    def convert_wave_to_spectogram(self, wave, sr, spectogram_kwargs = None):
        '''
        Gets a wave and its sample rate, then returns its stft magnitude

        params:
            wave (np.array) - wave to convert to stft
            sr (int) - sample rate of the wave
            spectogram_kwargs (dict) - kwargs for the stft conversion

        returns:
            np.ndarray - stft of the wave
        '''
        if not spectogram_kwargs: spectogram_kwargs = StftConverter_gl.STFT_KWARGS
        
        stft = lb.stft(y=wave, **spectogram_kwargs)
        mag, _ = lb.magphase(stft)

        return mag
    

    def convert_spectogram_to_wave(self, spectogram, sr, spectogram_kwargs=None):
        """
        Gets spectogram with stft magnitude, and converts to a waveform using Griffin-Lim algorithm.

        Args:
            spectogram (numpy.ndarray): The input spectrogram.
            sr (int): The sample rate of the waveform.
            spectogram_kwargs (dict, optional): Additional keyword arguments for the spectrogram computation.
                Defaults to None.

        Returns:
            numpy.ndarray: The reconstructed waveform.

        """
        if not spectogram_kwargs:
            spectogram_kwargs = StftConverter_gl.STFT_KWARGS

        griffinlim_kwargs = self._get_griffinlim_kwargs(spectogram_kwargs)

        mag = spectogram

        wave = lb.griffinlim(mag, **griffinlim_kwargs)

        return wave

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


    def _get_griffinlim_kwargs(self, spectogram_kwargs = None):
        '''
        Gets the kwargs for the griffinlim algorithm

        returns:
            dict - kwargs for the griffinlim algorithm
        '''
        if not spectogram_kwargs: spectogram_kwargs = StftConverter_gl.STFT_KWARGS

        return {
            'n_iter': 32,
            'pad_mode': 'constant',
            'momentum': 0.99,
            'init': 'random',
            'random_state': None,

            'hop_length': spectogram_kwargs['hop_length'],
            'win_length': spectogram_kwargs['win_length'],
            'window': spectogram_kwargs['window'],
            'center': spectogram_kwargs['center'],

            'n_fft':None,
            'length':None
        }
