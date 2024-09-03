import os
import re
import shutil

from visitor import Visitor, File_Delegate_visitor, Delegate_visitor
from process_dir_content import process_dir_content


class Get_csvs_visitor(Visitor):
    def __init__(self):
        self.csv_paths = []
    def file_action(self, path: str):
        if path.endswith(".csv"):
            self.csv_paths.append(path)

class Log_visitor(Visitor):
    def __init__(self, dict, csv_file, sample_name, dest_path):
        self.dict = dict
        self.csv_file = csv_file

        self.dest_path = dest_path
        self.sample_name = sample_name.split(".")[0]
        self.counter = 1
    def file_predicate(self, path: str) -> bool:
        return path.lower().endswith(".wav")
    def file_action(self, path: str):
        name = os.path.basename(path)
        name = name.split("_____")[1]

        new_name = f"{self.sample_name}_{self.counter}.wav"

        self.counter += 1

        rel_path = self.dict[name]

        self.csv_file.write(f"{new_name}|{rel_path}")
        self.csv_file.flush()

        shutil.copyfile(path, os.path.join(self.dest_path, new_name))

def connect_csv_files(source_path, sample_name, dest_path):
    dest_path_samples = os.path.join(dest_path, f"{sample_name}_samples")

    if not os.path.exists(dest_path_samples):
        os.mkdir(dest_path_samples)

    visitor = Get_csvs_visitor()
    process_dir_content(source_path, visitor)
    csv_paths = visitor.csv_paths

    dict = {}

    for path in csv_paths:
        with open(path, "r") as csv_f:
            lines = csv_f.readlines()
            for line in lines:
                line = line.split(";")
                name = line[0]
                rel_path = line[1]
                if len(line) > 2:
                    for i in range(2,len(line)):
                        rel_path += ";" + line[i]
                dict[name] = rel_path

    with open(os.path.join(dest_path, f"{sample_name}_paths.csv"), "w") as csv_file:
        csv_file.write("sample_name|relative_path_in_original_dataset\n")
        visitor = Log_visitor(dict, csv_file, sample_name, dest_path_samples)
        process_dir_content(source_path, visitor)

    print(f"Done, final count of {sample_name} samples: {visitor.counter - 1}")


def main():
    SAMPLE_NAME = r"kick"
    SOURCE_PATH = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots-not_ready\kick"
    DEST_PATH = fr"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots\{SAMPLE_NAME}"

    if not os.path.exists(DEST_PATH):
        os.mkdir(DEST_PATH)

    connect_csv_files(SOURCE_PATH, SAMPLE_NAME, DEST_PATH)

if __name__ == '__main__':
    main()
