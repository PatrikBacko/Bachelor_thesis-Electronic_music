import librosa as lb
import numpy as np


from src.VAE.utils.converters.converter import Converter


class StftConverter(Converter):
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
        if not kwargs: kwargs = StftConverter.STFT_KWARGS

        return {
            'type': 'stft',
            'kwargs': kwargs,
            'channels': 2,
            'height': kwargs['n_fft'] // 2 + 1
        }

    
    def convert_wave_to_spectrogram(self, wave, sr, spectrogram_kwargs = None):
        '''
        Gets a wave and its sample rate, then returns its stft magnitude and phase as angle in 2 channels

        params:
            wave (np.array) - wave to convert to stft
            sr (int) - sample rate of the wave
            spectrogram_kwargs (dict) - kwargs for the stft conversion

        returns:
            np.ndarray - stft of the wave
        '''
        if not spectrogram_kwargs: spectrogram_kwargs = StftConverter.STFT_KWARGS
        
        stft = lb.stft(y=wave, **spectrogram_kwargs)
        mag, phase = lb.magphase(stft)
        phase_angle = np.angle(phase)
        spectrogram = np.concatenate([mag.reshape(1,*mag.shape), phase_angle.reshape(1, *phase_angle.shape)], axis=0)

        return spectrogram
    

    def convert_spectrogram_to_wave(self, spectrogram, sr, spectrogram_kwargs = None):
        '''
        Gets a spectrogram and its sample rate, then returns its wave

        params:
            spectrogram (np.array) - spectrogram to convert to wave
            sr (int) - sample rate of the wave
            inverse_kwargs (dict) - kwargs for the inverse stft conversion

        returns:
            np.ndarray - wave of the spectrogram
        '''
        if not spectrogram_kwargs: spectrogram_kwargs = StftConverter.STFT_KWARGS

        inverse_stft_kwargs = self._get_inverse_stft_kwargs(spectrogram_kwargs)

        mag, phase_angle = spectrogram[0, :, :], spectrogram[1, :, :]
        phase = np.exp(1.j * phase_angle)
        stft = mag * phase
        wave = lb.istft(stft, **inverse_stft_kwargs)

        return wave

    def pad_or_trim_spectrogram(self, spectrogram, length):
        '''
        Pads or trims the spectrogram to the desired length

        params:
            spectrogram (np.ndarray) - spectrogram to pad or trim
            length (int) - desired length
            conversion_config (dict) - configuration for the conversion

        returns:
            np.ndarray - padded or trimmed spectrogram
        '''
        if spectrogram.shape[2] > length:
            spectrogram =  spectrogram[:, :, :length]
        else:
            last_column = spectrogram[:, :, -1:]
            
            padding = np.repeat(np.zeros_like(last_column), length - spectrogram.shape[2], axis=2)

            spectrogram = np.concatenate((spectrogram, padding), axis=2)

        return spectrogram

    
    def _get_inverse_stft_kwargs(self, spectrogram_kwargs = None):
        '''
        Gets the inverse stft kwargs from the spectrogram kwargs

        params:
            spectrogram_kwargs (dict) - kwargs for the stft conversion

        returns:
            dict - inverse stft kwargs
        '''
        if not spectrogram_kwargs: spectrogram_kwargs = StftConverter.STFT_KWARGS

        return {
            'hop_length': spectrogram_kwargs['hop_length'], 
            'win_length': spectrogram_kwargs['win_length'], 
            'window': spectrogram_kwargs['window'], 
            'center': spectrogram_kwargs['center'], 
            
            'n_fft':None,
            'length':None}
            
        