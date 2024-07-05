import os
import sys
import re
import shutil

from visitor import Visitor, File_Delegate_visitor, Delegate_visitor
from process_dir_content import process_dir_content
from remove_dir_with_content import remove_dir_with_content
from remove_MACOSX_dirs_visitor import Remove_MACOSX_dirs_visitor


class Sort_rename_log_visitor(Visitor):
    def __init__(self, source_path, dest_path: str, regex_pattern: str, sample_name: str = None, csv_file = None, copy: bool = True):
        self.source_path = source_path
        self.dest_path = dest_path
        self.regex_pattern = regex_pattern

        self.one_shot_name = sample_name
        self.csv_file = csv_file
        self.counter = 0

        self.copy = copy
    
    def file_action(self, path: str):
        name = os.path.basename(path)
        
        if re.search(self.regex_pattern, name, re.IGNORECASE) and name.lower().endswith(".wav"):
            self.counter += 1
            new_name = f"{self.one_shot_name}_{self.counter}.wav"

            if self.copy == True:
                shutil.copy(path, os.path.join(self.dest_path, f"{name}_____{new_name}"))
            else:
                shutil.move(path, os.path.join(self.dest_path, f"{name}_____{new_name}"))
            
            if self.csv_file != None:
                rel_path = os.path.relpath(path, start=self.source_path)
                self.csv_file.write(f"{new_name};{rel_path}\n")



class Sort_with_regex_visitor(Visitor):
    def __init__(self, dest_path: str, regex_pattern: str):
        self.dest_path = dest_path
        self.regex_pattern = regex_pattern
    
    def file_action(self, path: str):
        name = os.path.basename(path)

        if re.search(self.regex_pattern, name, re.IGNORECASE):
                shutil.move(path, os.path.join(self.dest_path, name))



class Sort_with_size_visitor(Visitor):
    def __init__(self, dest_path, size: int):
        self.dest_path = dest_path
        self.size = size
    def file_action(self, path: str):
        name = os.path.basename(path)
        size = os.path.getsize(path)
        if size > self.size:
            shutil.move(path, os.path.join(self.dest_path, name))


def sort_samples(start_path, source_path, dest_path, sample_name, regex_pattern):
    dest_path_sure = os.path.join(dest_path, "almost_sure")
    dest_path_manual = os.path.join(dest_path, "manual_check")
    dest_path_loops = os.path.join(dest_path, "loops")

    csv_file_path = os.path.join(dest_path, f"{sample_name}.csv")

    paths = [dest_path, dest_path_sure, dest_path_manual, dest_path_loops]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    with open(csv_file_path, "w", encoding="utf-8") as csv_file:
        visitor = Sort_rename_log_visitor(source_path, dest_path_sure, regex_pattern, sample_name, csv_file, copy=False)
        process_dir_content(start_path, visitor)

    #maybe add other words to regex_pattern that could signal that a file is a loop
    visitor = Sort_with_regex_visitor(dest_path_loops, "loop")
    process_dir_content(dest_path_sure, visitor)

    visitor = Sort_with_size_visitor(dest_path_manual, 1_000_000)
    process_dir_content(dest_path_sure, visitor)


def remove_not_wav_files_and_files(path):
    def dir_action(path):
        if len(os.listdir(path)) < 1:
            os.chmod(path, 0o777)
            os.rmdir(path)

    #remove dirs without .wav files
    visitor = Delegate_visitor(lambda path: not path.lower().endswith(".wav"),
                               lambda path: os.remove(path),
                               lambda path: True,
                               dir_action)

    process_dir_content(path, visitor)

def remove_loop_dirs(path):
    class Visitor_4(Visitor):
        def file_action(self, path: str):
            name = os.path.basename(path)
            if re.match(r"loop", name, re.IGNORECASE):
                os.remove(path)
        def dir_action(self, path):
            if len(os.listdir(path)) < 1:
                os.chmod(path, 0o777)
                os.rmdir(path)
    
    process_dir_content(path, Visitor_4())

def main():
    SAMPLE_NAME = "other"
    REGEX_PATTERN = r".*"
    SOURCE_PATH = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\unzipped-samples"
    START_PATH = SOURCE_PATH
    DEST_PATH = fr"C:\Users\llama\Desktop\programming shit\Bakalarka\{SAMPLE_NAME}_samples"

    sort_samples(START_PATH, SOURCE_PATH, DEST_PATH, SAMPLE_NAME, REGEX_PATTERN)

    remove_not_wav_files_and_files(SOURCE_PATH)

    # path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\kick_samples\manual_check"
    # process_dir_content(path, File_Delegate_visitor(lambda path: os.path.getsize(path) > 1_000_000, 
    #                                                 lambda path: os.remove(path)))
    
    
if __name__ == "__main__":
    main()