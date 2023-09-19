import os
import re 
import shutil

from visitor import Visitor
from process_dir_content import process_dir_content


class Logger_visitor(Visitor):
    def __init__(self, samples_set: set, csv_file):
        self.samples_set = samples_set
        self.csv_file = csv_file
    def file_action(self, path: str):
        name = os.path.basename(path)
        if name in self.samples_set:
            self.csv_file.write(f"{name};{path}\n")
    def file_predicate(self, path: str) -> bool:
        if path.lower().endswith(".wav"):
            return True
        return False


def log_samples(samples_path, data_path, csv_file_path):
    samples_set = set(os.listdir(samples_path))

    with open(csv_file_path, "w", encoding="utf-8") as csv_file:
        process_dir_content(data_path, Logger_visitor(samples_set, csv_file))


def main():
    samples_path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots\snare"
    # data_path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\backup-unzipped-samples"
    # csv_file_path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots\snare.csv"

    # log_samples(samples_path, data_path, csv_file_path)
  
    hashset = set()

    for file in os.listdir(samples_path):
        if file in hashset:
            print(file)
        else:
            hashset.add(file)
    

if __name__ == "__main__":
    main()