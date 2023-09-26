import pandas as pd

SAMPLE_NAME = "kick"
PATH = Fr"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots\{SAMPLE_NAME}\{SAMPLE_NAME}_genres.csv"

dataset = pd.read_csv(PATH, sep="|")

#print count of samples for each genre
print(dataset["name"][dataset["genre"] == "House" ].count())