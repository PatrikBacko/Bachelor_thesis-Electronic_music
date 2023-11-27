from typing import Any
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


class Wave:
    def __init__(self, wav_file_path: str) -> None:
        self.load_wave(wav_file_path)

    def load_wave(self, wav_file_path: str) -> None:
        self.wave_info = sf.info(wav_file_path)
        self.wave, self.sample_rate = lb.load(wav_file_path, sr=None)

    def play_wave(self) -> None:
        # initialize PyAudio
        p = pyaudio.PyAudio()

        # open a stream
        stream = p.open(format=pyaudio.paFloat32,
                    channels=self.wave_info.channels,
                    rate=self.wave_info.samplerate,
                    output=True)

        # play audio
        stream.write(self.wave.tobytes())

        # stop stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

    def show_wave(self) -> None:
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(self.wave, sr=self.sample_rate)
        plt.show()

    def save(self, path: str) -> None:
        sf.write(path, self.wave, self.sample_rate) 

    def to_Mel(self):
        return MelSpectogram(self)
    

class MelSpectogram_parameters:
        def __init__(self, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0) -> None:
            self.S = S
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length
            self.window = window
            self.center = center
            self.pad_mode = pad_mode

class MelSpectogram:
    def __init__(self, wave: Wave) -> None:
        self.from_wave(wave)

    def from_wave(self, wave: Wave, mel_parameters=MelSpectogram_parameters()) -> None:
        self.mel_spectogram = lb.feature.melspectrogram(wave.wave, sr=wave.sample_rate,
                                                        n_fft=mel_parameters.n_fft,
                                                        hop_length=mel_parameters.hop_length,
                                                        win_length=mel_parameters.win_length,
                                                        window=mel_parameters.window,
                                                        center=mel_parameters.center,
                                                        pad_mode=mel_parameters.pad_mode,
                                                        power=mel_parameters.power)
        self.wave_info = wave.wave_info
        self.mel_parameters = mel_parameters

    def show_plot(self) -> None:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(self.mel_spectogram, x_axis='time')
        plt.colorbar()
        plt.title('Mel Spectogram')
        plt.tight_layout()
        plt.show()

    
class Mfcc_parameters:
    def __init__(self, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0) -> None:
        self.S = S
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.lifter = lifter
 
class Mfcc:
    def __init__(self) -> None:
        pass
    
    def from_wave(self, wave: Wave, mfcc_parameters = Mfcc_parameters()) -> None:
        self.mfcc = lb.feature.mfcc(wave.wave, sr=wave.sample_rate,
                                    n_mfcc=mfcc_parameters.n_mfcc,
                                    dct_type=mfcc_parameters.dct_type,
                                    norm=mfcc_parameters.norm,
                                    lifter=mfcc_parameters.lifter)
        self.wave_info = wave.wave_info
        self.mfcc_parameters = mfcc_parameters
        self.mel_parameters = MelSpectogram_parameters()

    def from_mel(self, melSpectogram, mel_parameters=MelSpectogram_parameters(), mfcc_parameters = Mfcc_parameters()) -> None:
        self.mfcc = lb.feature.mfcc(S=melSpectogram.mel_spectogram, sr=melSpectogram.wave_info.samplerate,
                                    n_mfcc=mfcc_parameters.n_mfcc,
                                    dct_type=mfcc_parameters.dct_type,
                                    norm=mfcc_parameters.norm,
                                    lifter=mfcc_parameters.lifter)
        self.wave_info = melSpectogram.wave_info
        self.mfcc_parameters = mfcc_parameters
        self.mel_parameters = mel_parameters
    
    def to_wave(self) -> Wave:
        wave = Wave(None)
        wave.wave = lb.feature.inverse.mfcc_to_audio(self.mfcc, sr=self.wave_info.samplerate,
                                                    n_fft=self.mel_parameters.n_fft,
                                                    hop_length=self.mel_parameters.hop_length,
                                                    win_length=self.mel_parameters.win_length,
                                                    window=self.mel_parameters.window,
                                                    center=self.mel_parameters.center,
                                                    pad_mode=self.mel_parameters.pad_mode,
                                                    power=self.mel_parameters.power,
                                                    )
        wave.sample_rate = self.wave_info.samplerate
        return wave

    def show_plot(self) -> None:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(self.mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()