import os
import shutil


path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots\kick\kick_paths.csv"

prev_number = 0
with open(path, "r") as csv_f:
    lines = csv_f.readlines()
    for line in lines:
        if line == "sample_name;relative_path_in_original_dataset\n":
            continue
        line = line.split(";")
        number = int(line[0].split("_")[1].split(".")[0])
        if number - prev_number != 1:
            print(number)
        prev_number = number