import os
import django

path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots"

print(list(map(lambda path: os.path.basename(path), os.listdir(path))))