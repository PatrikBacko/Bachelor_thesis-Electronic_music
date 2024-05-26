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
    
    def convert_wave_to_spectrogram(self, wave, sr, spectrogram_kwargs = None):
        '''
        Gets a wave and its sample rate, then returns its cqt

        params:
            wave (np.array) - wave to convert to cqt
            sr (int) - sample rate of the wave
            spectrogram_kwargs (dict) - kwargs for the cqt conversion

        returns:
            np.ndarray - cqt of the wave
        '''

        if not spectrogram_kwargs: spectrogram_kwargs = CqtConverter.CQT_KWARGS
        
        cqt = lb.cqt(y= wave, sr = sr, **spectrogram_kwargs)
        mag, phase = lb.magphase(cqt)

        phase_angle = np.angle(phase)

        spectrogram = np.concatenate([mag.reshape(1,*mag.shape), phase_angle.reshape(1, *phase_angle.shape)], axis=0)

        return spectrogram
    

    def convert_spectrogram_to_wave(self, spectrogram, sr, spectrogram_kwargs = None):
        '''
        Gets a spectrogram and its sample rate, then returns its wave

        params:
            spectrogram (np.array) - spectrogram to convert to wave
            sr (int) - sample rate of the wave
            inverse_kwargs (dict) - kwargs for the inverse cqt conversion

        returns:
            np.ndarray - wave of the spectrogram
        '''
        if not spectrogram_kwargs: spectrogram_kwargs = CqtConverter.CQT_KWARGS

        inverse_cqt_kwargs = self._get_inverse_cqt_kwargs(spectrogram_kwargs)

        # mag, phase_angle = np.split(spectrogram, 2, axis=0)
        mag, phase_angle = spectrogram[0, :, :], spectrogram[1, :, :]

        print(mag.shape, phase_angle.shape)
        phase = np.exp(1.j * phase_angle)
        print(phase.shape)
        cqt = mag * phase
        wave = lb.icqt(C = cqt, sr = sr, **inverse_cqt_kwargs)

        return wave
    
    def pad_or_trim_spectrogram(self, spectrogram, length):
        raise NotImplementedError("CQT does not support padding or trimming")

    def _get_inverse_cqt_kwargs(self, spectrogram_kwargs = None):
        '''
        Gets the inverse cqt kwargs from the spectrogram kwargs

        params:
            spectrogram_kwargs (dict) - kwargs for the cqt conversion

        returns:
            dict - kwargs for the inverse cqt conversion
        '''
        if not spectrogram_kwargs: spectrogram_kwargs = CqtConverter.CQT_KWARGS

        return {
        'hop_length': spectrogram_kwargs['hop_length'],
        'fmin': spectrogram_kwargs['fmin'],
        'bins_per_octave': spectrogram_kwargs['bins_per_octave'],
        'tuning': spectrogram_kwargs['tuning'],
        'filter_scale': spectrogram_kwargs['filter_scale'],
        'norm': spectrogram_kwargs['norm'],
        'sparsity': spectrogram_kwargs['sparsity'],
        'window': spectrogram_kwargs['window'],
        'scale': spectrogram_kwargs['scale'],
        'res_type': spectrogram_kwargs['res_type'],

        'length' : None

        }