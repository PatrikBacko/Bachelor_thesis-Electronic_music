import pandas as pd

def get_data_info(csv_path: str, genres: list[str], sample_type: str):
    dataset = pd.read_csv(csv_path, sep="|")
    print(f"### {sample_type}")
    print()
    print("|Genre|Count|")
    print("|------|------|")
    for genre in genres:
        genre_dataset = dataset[dataset["genre"] == genre]
        print(f"|{genre}|{len(genre_dataset)}|")
    

def main():
    genres = [
        "DnB",
        "Dubstep",
        "EDM",
        "Techno",
        "Trap",
        "House",
        "Trance",
        "Hip-Hop",
        "Funk",
        "Lofi",
        "Unclassified",
        ]
    samples_types = ['clap', 'crash', 'cymbal', 'hat', 'kick', 'other', 'ride', 'snare', 'tom']

    for sample_type in samples_types:
        path = fr"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\drums-one_shots\{sample_type}\{sample_type}_genres.csv"
        get_data_info(path, genres, sample_type)
    print("\n")

if __name__ == "__main__":
    main()