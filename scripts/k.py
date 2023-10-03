import os

PATH = r'C:\\Users\\llama\Desktop\\programming shit\Bakalarka\\temp'
CSV_PATH = r'.\\Bakalaris-data\\temp.csv'
site = 'https://www.waproduction.com/sounds/items/free?keyword=&types%5B2%5D=1&instruments%5B3%5D=1&filter-prices-range=&price_changed_from=0&price_changed_to=0&filter=1'


# Get the list of all files from PATH
files = os.listdir(PATH)

#open the csv file for writing the file names
with open(CSV_PATH, 'w') as f:
    # Iterate over the list of files and write each one
    for file in files:
        f.write(file + ';' + site + '\n')

