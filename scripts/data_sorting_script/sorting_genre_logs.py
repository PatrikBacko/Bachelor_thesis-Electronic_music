import os
import re
import shutil

from visitor import Visitor
from process_dir_content import process_dir_content

def main():
    csv_path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\samples_genre.csv"
    samples_path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots"

    type_dict = {}
    type_dict["snare"] = {}
    type_dict["kick"] = {}
    type_dict["hat"] = {}
    type_dict["crash"] = {}
    type_dict["clap"] = {}
    type_dict["cymbal"] = {}
    type_dict["ride"] = {}
    type_dict["tom"] = {}
    type_dict["other"] = {}

    with open(csv_path, "r", encoding="utf-8") as csv_file:
        csv_file.readline()
        for line in csv_file:
            name, genre = line.split("|")
            type_dict[name.split("_")[0]][int(name.split("_")[1][:-4])] = (name, genre[:-1])

    for key in type_dict.keys():

        num_of_samples = len(os.listdir(os.path.join(samples_path, key, f"{key}_samples")))

        for i in range(1, num_of_samples + 1):
            if not i in type_dict[key].keys():
                type_dict[key][i] = ((f"{key}_{i}.wav", "None"))


        samples_list = list(type_dict[key].values())
        samples_list.sort(key=lambda x: (int(x[0].split("_")[1][:-4]), x[1]))

        with open(f"{os.path.join(samples_path, key, key)}_genres.csv", "w", encoding="utf-8") as csv_f:
            csv_f.write("name|genre\n")

            for name, genre in samples_list:
                csv_f.write(f"{name}|{genre}\n")

            
if __name__ == "__main__":
    main()