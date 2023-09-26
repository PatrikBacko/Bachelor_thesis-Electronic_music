import pandas as pd
import os
import shutil

from visitor import Visitor 
from process_dir_content import process_dir_content


class Rename_wav_visitor(Visitor):
    def file_action(self, path: str):
        if path.endswith(".wav"):
            name = os.path.basename(path)
            sample_type, number = name[:-4].split("_")
            new_name = f"{sample_type}_{number.zfill(4)}.wav"
            new_path = os.path.join(os.path.dirname(path), new_name)
            os.rename(path, new_path)

class Rename_csv_visitor(Visitor):
    def file_action(self, path: str):
        if path.endswith(".csv"):
            with open(path, "r", encoding="utf-8") as csv_file:
                first_line = csv_file.readline()
                lines = csv_file.readlines()
            with open(path, "w", encoding="utf-8") as new_csv_file:
                new_csv_file.write(first_line)
                for line in lines:
                    name, other = line.split("|")
                    sample_type, number = name[:-4].split("_")
                    new_name = f"{sample_type}_{number.zfill(4)}.wav"
                    new_csv_file.write(f"{new_name}|{other}")

class Rename_genres_visitor(Visitor):
    def file_action(self, path: str):
        if path.endswith("genres.csv"):
            with open(path, "r", encoding="utf-8") as csv_file:
                first_line = csv_file.readline()
                lines = csv_file.readlines()
            with open(path, "w", encoding="utf-8") as new_csv_file:
                new_csv_file.write(first_line)
                for line in lines:
                    name, genre = line.split("|")
                    if genre == "Drum&Bass\n":
                        genre = "DnB\n"
                    new_csv_file.write(f"{name}|{genre}")


def rename_wavs(path: str):
    visitor = Rename_wav_visitor()
    process_dir_content(path, visitor)

def rename_csvs(path: str):
    visitor = Rename_csv_visitor()
    process_dir_content(path, visitor)

def rename_genres(path: str):
    visitor = Rename_genres_visitor()
    process_dir_content(path, visitor)

def main():
    path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\drums-one_shots"
    rename_genres(path)
    # rename_csvs(path)
    # rename_wavs(path)

if __name__ == "__main__":
    main()

            