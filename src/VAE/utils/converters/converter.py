from abc import ABC, abstractmethod 


class Converter(ABC):

   @abstractmethod
   def convert_spectrogram_to_wave(self, spectrogram, sr, spectrogram_kwargs):
      '''
      Converts a spectrogram to wave

      params:
         spectrogram (np.ndarray) - mfcc to convert
         sr (int) - sample rate of the wave
         spectrogram_kwargs (dict) - kwargs for the inverse spectrogram

      returns:
         np.array - wave      

      raises:
         InvalidSamplingException - if there is an error with spectrogram inverse conversion
      '''
      ...

   @abstractmethod
   def convert_wave_to_spectrogram(self, wave, sr, spectrogram_kwargs):
      '''
      Converts a wave to spectrogram

      params:
         wave (np.array) - wave to convert to spectrogram
         sr (int) - sample rate of the wave
         spectrogram_kwargs (dict) - kwargs for the spectrogram

      returns:
         np.ndarray - spectrogram
      '''
      ... 

   @abstractmethod
   def get_default_config(self, conversion_type, pad_or_trim_length):
      '''
      Gets the default kwargs and model configuration for the conversion type
      
      params:
         conversion_type (str) - type of conversion
         pad_or_trim_length (int) - length to pad or trim the spectrogram

      returns:
         dict - default kwargs for the conversion
      '''
      ... 

   @abstractmethod
   def pad_or_trim_spectrogram(self, spectrogram, length):
      '''
      Pads or trims the spectrogram to the desired length

      params:
         conversion_config (dict) - configuration for the conversion

      returns:
         np.ndarray - padded or trimmed spectrogram
      '''
      ...
