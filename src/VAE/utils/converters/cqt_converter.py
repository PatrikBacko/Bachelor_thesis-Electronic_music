import librosa as lb
import numpy as np

from src.VAE.utils.converters.converter import Converter


class CqtConverter(Converter):
    CQT_KWARGS = {
        'hop_length':256,
        'fmin': None,
        'n_bins':512,
        'bins_per_octave':96,
        'tuning':0.0,
        'filter_scale':1,
        'norm':1,
        'sparsity':0.01,
        'window':'hann',
        'scale':True,
        'pad_mode':'constant',
        'res_type':'soxr_hq'
    }

    def get_default_config(self):
        '''
        Gets the default config for the cqt conversion

        returns:
            dict - default config for the cqt conversion
        '''
        return {
            'type': 'cqt',
            'kwargs': CqtConverter.CQT_KWARGS,
            'channels': 2,
            'height': CqtConverter.CQT_KWARGS['n_bins'],
        }
    
    def convert_wave_to_spectogram(self, wave, sr, spectogram_kwargs = None):
        '''
        Gets a wave and its sample rate, then returns its cqt

        params:
            wave (np.array) - wave to convert to cqt
            sr (int) - sample rate of the wave
            spectogram_kwargs (dict) - kwargs for the cqt conversion

        returns:
            np.ndarray - cqt of the wave
        '''

        if not spectogram_kwargs: spectogram_kwargs = CqtConverter.CQT_KWARGS
        
        cqt = lb.cqt(y= wave, sr = sr, **spectogram_kwargs)
        mag, phase = lb.magphase(cqt)

        phase_angle = np.angle(phase)

        spectogram = np.concatenate([mag.reshape(1,*mag.shape), phase_angle.reshape(1, *phase_angle.shape)], axis=0)

        return spectogram
    

    def convert_spectogram_to_wave(self, spectogram, sr, spectogram_kwargs = None):
        '''
        Gets a spectogram and its sample rate, then returns its wave

        params:
            spectogram (np.array) - spectogram to convert to wave
            sr (int) - sample rate of the wave
            inverse_kwargs (dict) - kwargs for the inverse cqt conversion

        returns:
            np.ndarray - wave of the spectogram
        '''
        if not spectogram_kwargs: spectogram_kwargs = CqtConverter.CQT_KWARGS

        inverse_cqt_kwargs = self._get_inverse_cqt_kwargs(spectogram_kwargs)

        # mag, phase_angle = np.split(spectogram, 2, axis=0)
        mag, phase_angle = spectogram[0, :, :], spectogram[1, :, :]

        print(mag.shape, phase_angle.shape)
        phase = np.exp(1.j * phase_angle)
        print(phase.shape)
        cqt = mag * phase
        wave = lb.icqt(C = cqt, sr = sr, **inverse_cqt_kwargs)

        return wave

    def _get_inverse_cqt_kwargs(self, spectogram_kwargs = None):
        '''
        Gets the inverse cqt kwargs from the spectogram kwargs

        params:
            spectogram_kwargs (dict) - kwargs for the cqt conversion

        returns:
            dict - kwargs for the inverse cqt conversion
        '''
        if not spectogram_kwargs: spectogram_kwargs = CqtConverter.CQT_KWARGS

        return {
        'hop_length': spectogram_kwargs['hop_length'],
        'fmin': spectogram_kwargs['fmin'],
        'bins_per_octave': spectogram_kwargs['bins_per_octave'],
        'tuning': spectogram_kwargs['tuning'],
        'filter_scale': spectogram_kwargs['filter_scale'],
        'norm': spectogram_kwargs['norm'],
        'sparsity': spectogram_kwargs['sparsity'],
        'window': spectogram_kwargs['window'],
        'scale': spectogram_kwargs['scale'],
        'res_type': spectogram_kwargs['res_type'],

        'length' : None

        }