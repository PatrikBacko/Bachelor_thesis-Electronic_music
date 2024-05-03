from abc import ABC, abstractmethod 


class Converter(ABC):

   @abstractmethod
   def convert_spectogram_to_wave(self, spectogram, sr, spectogram_kwargs):
      '''
      Converts a spectogram to wave

      params:
         spectogram (np.ndarray) - mfcc to convert
         sr (int) - sample rate of the wave
         spectogram_kwargs (dict) - kwargs for the inverse spectogram

      returns:
         np.array - wave      

      raises:
         InvalidSamplingException - if there is an error with spectogram inverse conversion
      '''
      ...

   @abstractmethod
   def convert_wave_to_spectogram(self, wave, sr, spectogram_kwargs):
      '''
      Converts a wave to spectogram

      params:
         wave (np.array) - wave to convert to spectogram
         sr (int) - sample rate of the wave
         spectogram_kwargs (dict) - kwargs for the spectogram

      returns:
         np.ndarray - spectogram
      '''
      ... 

   @abstractmethod
   def get_default_config(self, conversion_type, pad_or_trim_length):
      '''
      Gets the default kwargs and model configuration for the conversion type
      
      params:
         conversion_type (str) - type of conversion

      returns:
         dict - default kwargs for the conversion
      '''
      ... 
