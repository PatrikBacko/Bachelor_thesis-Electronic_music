import threading
import os
import re
import shutil
import time

import pyaudio
import wave

class Cancel_token:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def is_cancelled(self):
        return self.cancelled

class AudioFile:
    chunk = 1024
    # playing = False

    def __init__(self, file, cancel_token):
        """ Init audio stream """ 
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )
        self.cancel_token = cancel_token

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        # self.playing = True
        while data != b'' and not self.cancel_token.is_cancelled():
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)
        # self.playing = False

    def close(self):
        """ Graceful shutdown """ 
        self.stream.close()
        self.p.terminate()

    # def stop(self):
    #     """Stop playback"""
    #     self.playing = False
    #     self.close()


def check_audio_file(path) -> int:

    def except_func(e: Exception) -> int:
        print(f"Error while playing audio file: {path}, error: {e}")
        return 2
    
    cancel_token = Cancel_token()
    try:
        audio_file = AudioFile(path, cancel_token)
    except Exception as e:
        return except_func(e)
    
    t = threading.Thread(target=audio_file.play)
    try:
        t.start()
        is_sample = input()
        cancel_token.cancel()
        audio_file.close()
        t.join()

        if is_sample == "n" or is_sample == "3":
            return 1
        elif is_sample == "r" or is_sample == "0":
            return check_audio_file(path)
        elif is_sample == "e" or is_sample == "9":
            return except_func(Exception("User was not sure"))
        else:
            return 0
    except Exception as e:
        t.join()
        return except_func(e)
    

def manual_check(source_path):
    not_sample_path = os.path.join(source_path, "not_sample")
    if not os.path.exists(not_sample_path):
        os.mkdir(not_sample_path)

    error_files_path = os.path.join(source_path, "error_files")
    if not os.path.exists(error_files_path):
        os.mkdir(error_files_path)

    controlled_samples_path = os.path.join(source_path, "controlled_samples")
    if not os.path.exists(controlled_samples_path):
        os.mkdir(controlled_samples_path)

    for file in os.listdir(source_path):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(source_path, file)
        return_value = check_audio_file(path)

        if return_value == 0:
            shutil.move(path, os.path.join(controlled_samples_path, file))
        if return_value == 1:
            shutil.move(path, os.path.join(not_sample_path, file))
        elif return_value == 2:
            shutil.move(path, os.path.join(error_files_path, file))


def main():
    source_path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\manual_check"
    manual_check(source_path)

if __name__ == "__main__":
    main()