import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import wave 
import soundfile as sf


file_path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots\kick\kick_samples\kick_0001.wav"

info = sf.info(file_path)
print(f"Channels: {info.channels}")
print(f"Sample rate: {info.samplerate} Hz")
print(f"Duration: {info.duration} seconds")

y, _ = librosa.load(file_path, sr=info.samplerate)

duration = librosa.get_duration(y=y, sr=info.samplerate)
print(f"Librosa duration: {duration} seconds")


librosa.display.waveshow(y, sr=info.samplerate, label="Hat_0008")
plt.show()

x= "x"