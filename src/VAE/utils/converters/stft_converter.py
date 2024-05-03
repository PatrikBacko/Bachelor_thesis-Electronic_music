import librosa as lb
import numpy as np


from src.VAE.utils.converters.converter import Converter


class StftConverter(Converter):
    STFT_KWARGS = {
     'n_fft':2048,
     'hop_length':None,
     'win_length':None,
     'window':'hann',
     'center':True,
     'dtype':None,
     'pad_mode':'constant',

    }

    
    def get_default_config(self):
        '''
        Gets the default config for the stft conversion

        returns:
            dict - default config for the stft conversion
        '''
        return {
            'type': 'stft',
            'kwargs': StftConverter.STFT_KWARGS,
            'channels': 2,
            'height': StftConverter.STFT_KWARGS['n_fft'] // 2 + 1,
        }

    
    def convert_wave_to_spectogram(self, wave, sr, spectogram_kwargs = None):
        '''
        Gets a wave and its sample rate, then returns its stft

        params:
            wave (np.array) - wave to convert to stft
            sr (int) - sample rate of the wave
            spectogram_kwargs (dict) - kwargs for the stft conversion

        returns:
            np.ndarray - stft of the wave
        '''
        if not spectogram_kwargs: spectogram_kwargs = StftConverter.STFT_KWARGS
        
        stft = lb.stft(y=wave, **spectogram_kwargs)
        mag, phase = lb.magphase(stft)
        phase_angle = np.angle(phase)
        spectogram = np.concatenate([mag.reshape(1,*mag.shape), phase_angle.reshape(1, *phase_angle.shape)], axis=0)

        return spectogram
    

    def convert_spectogram_to_wave(self, spectogram, sr, spectogram_kwargs = None):
        '''
        Gets a spectogram and its sample rate, then returns its wave

        params:
            spectogram (np.array) - spectogram to convert to wave
            sr (int) - sample rate of the wave
            inverse_kwargs (dict) - kwargs for the inverse stft conversion

        returns:
            np.ndarray - wave of the spectogram
        '''
        if not spectogram_kwargs: spectogram_kwargs = StftConverter.STFT_KWARGS

        inverse_stft_kwargs = self._get_inverse_stft_kwargs(spectogram_kwargs)

        mag, phase_angle = spectogram[0, :, :], spectogram[1, :, :]
        phase = np.exp(1.j * phase_angle)
        stft = mag * phase
        wave = lb.istft(stft, **inverse_stft_kwargs)

        return wave

    
    def _get_inverse_stft_kwargs(self, spectogram_kwargs = None):
        '''
        Gets the inverse stft kwargs from the spectogram kwargs

        params:
            spectogram_kwargs (dict) - kwargs for the stft conversion

        returns:
            dict - inverse stft kwargs
        '''
        if not spectogram_kwargs: spectogram_kwargs = StftConverter.STFT_KWARGS

        return {
            'hop_length': spectogram_kwargs['hop_length'], 
            'win_length': spectogram_kwargs['win_length'], 
            'window': spectogram_kwargs['window'], 
            'center': spectogram_kwargs['center'], 
            
            'n_fft':None,
            'length':None}
            
        