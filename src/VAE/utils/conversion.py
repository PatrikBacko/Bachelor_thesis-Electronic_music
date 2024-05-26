from src.VAE.utils.converters.mfcc_converter import MfccConverter
from src.VAE.utils.converters.stft_converter import StftConverter
from src.VAE.utils.converters.cqt_converter import CqtConverter
from src.VAE.utils.converters.stft_converter_gl import StftConverter_gl
from src.VAE.utils.converters.stft_converter_gl_2_channels import StftConverter_gl_2_channels

CONVERTERS = {
    'mfcc': MfccConverter(),
    'stft': StftConverter(),
    'cqt': CqtConverter(),
    'stft_gl': StftConverter_gl(),
    'stft_gl_2_channels': StftConverter_gl_2_channels()

}

def _get_converter(conversion_type):
    '''
    Gets a conversion type and returns its converter

    params:
        conversion_type (str) - type of conversion

    returns:
        Converter - converter for the conversion type
    '''
    return CONVERTERS[conversion_type]

def get_converters_list():
    '''
    Gets a list of available converters

    returns:
        list - list of available converters
    '''
    return list(CONVERTERS.keys())


def get_default_conversion_config(conversion_type):
    '''
    Gets a conversion type and returns its default config (kwargs, model settings ...)

    params:
        conversion_type (str) - type of conversion

    returns:
        dict - default config for the conversion type
    '''
    return _get_converter(conversion_type).get_default_config()



def convert_wave_to_spectrogram(wave, sr, conversion_config):
    '''
    Gets a wave and its sample rate, then returns its spectrogram

    params:
        wave (np.array) - wave to convert to spectrogram
        sr (int) - sample rate of the wave
        conversion_config (dict) - config for the spectrogram conversion
    '''

    return _get_converter(conversion_config['type']).convert_wave_to_spectrogram(wave, sr, conversion_config['kwargs'])


def convert_spectrogram_to_wave(spectrogram, sr, conversion_config):
    '''
    Gets a spectrogram and returns invertly converted wave

    params:
        spectrogram (np.ndarray) - spectrogram to convert
        sr (int) - sample rate of the wave
        conversion_config (dict) - config for the spectrogram conversion
    '''

    return _get_converter(conversion_config['type']).convert_spectrogram_to_wave(spectrogram, sr, conversion_config['kwargs'])

def pad_or_trim(spectrogram, length, conversion_config):
    '''
    Pads or trims the spectrogram to the desired length

    params:
        spectrogram (np.ndarray) - spectrogram to pad or trim
        length (int) - length to pad or trim the spectrogram
        conversion_config (dict) - configuration for the conversion
    '''
    return _get_converter(conversion_config['type']).pad_or_trim_spectrogram(spectrogram, length)
