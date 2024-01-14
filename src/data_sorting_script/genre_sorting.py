import os 
import shutil
import re


from visitor import Visitor
from process_dir_content import process_dir_content


class Acoustic_samples_visitor(Visitor):
    def __init__(self, source_path: str, genre: str, genre_csv, dict: dict) -> None:
        self.genre_csv = genre_csv
        self.dict = dict
        self.genre = genre
        self.source_path = source_path
    def file_predicate(self, path: str) -> bool:
        return path.lower().endswith(".wav")
    def file_action(self, path: str) -> None:
        rel_path = os.path.relpath(path, self.source_path)

        if rel_path in self.dict.keys():
            new_name = self.dict[rel_path]
            self.genre_csv.write(f"{new_name}|{self.genre}\n") 


class Genre_sorting_visitor(Visitor):
    def __init__(self, source_path: str, genre: str, genre_csv, dict: dict) -> None:
        self.genre_csv = genre_csv
        self.dict = dict
        self.genre = genre
        self.source_path = source_path
    def file_predicate(self, path: str) -> bool:
        return path.lower().endswith(".wav")
    def file_action(self, path: str) -> None:
        rel_path = os.path.relpath(path, self.source_path)

        if rel_path in self.dict.keys():
            new_name = self.dict[rel_path]
            self.genre_csv.write(f"{new_name}|{self.genre}\n")


class Create_sorted_samples_dict_visitor(Visitor):
    def __init__(self, dict: dict) -> None:
        self.dict = dict
    def file_predicate(self, path: str) -> bool:
        return path.lower().endswith(".csv")
    def file_action(self, path: str):
        with open(path, "r") as csv_file:
            csv_file.readline()
            for line in csv_file:
                name, rel_path = line.split("|")
                self.dict[rel_path[:-1]] = name
        

def genre_sorting(source_path, start_path, genre, genre_csv_path, path_to_sorted_samples):

    if not os.path.exists(genre_csv_path):
        with open(genre_csv_path, "w") as genre_csv:
            genre_csv.write("name|genre\n")

    visitor = Create_sorted_samples_dict_visitor({})
    process_dir_content(path_to_sorted_samples, visitor)
    dict_of_samples = visitor.dict

    with open(genre_csv_path, "a+") as genre_csv:
        visitor = Genre_sorting_visitor(source_path, genre, genre_csv, dict_of_samples)
        process_dir_content(start_path, visitor)



def main():
    #genres = ["Hip-Hop", "House", "Techno", "Trance", "Trap", "Drum&Bass", "Dubstep", "EDM", ?"Funk", ??"Lofi"]
    GENRE = r"Acuostic"
    SOURCE_PATH = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\backup-unzipped-samples"
    START_PATH = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\backup-unzipped-samples"
    PATH_TO_SORTED_SAMPLES = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots"
    GENRE_CSV_PATH = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\samples_genre.csv"

    genre_sorting(SOURCE_PATH, START_PATH, GENRE, GENRE_CSV_PATH, PATH_TO_SORTED_SAMPLES)

if __name__ == "__main__":
    main()