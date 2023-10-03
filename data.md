# Drum one-shot samples Dataset
- Dataset contains thousands of electronic and acoustic drum one-shot samples. These one-shots are sorted by drum type and electronic genre. 
- Data are sorted into directories named after drum type of samples which are inside. Inside these directories are 2 ".csv" files and directory with ".wav" samples. 
  - First file is called "{drum_type}_paths.csv" and for each sample contains relative path to the original sample, in raw dataset.
  - Second file is called "{drum_type}_genres.csv" and for each sample contains its genre.
- There are also raw data, which are downloaded .zip directories from which the data were sorted.
  - raw data directory also contains "raw_data_origin.csv" files with links to sites from which the data were downloaded.

### Drum Types:
- kick
- snare
- clap
- tom
- crash
- cymbal
- hat
- ride
- other

### Genres:
- DnB
- Dubstep
- EDM
- Techno
- Trap
- House
- Trance
- Hip-Hop
- Funk
- Lofi
- Unclassified

### Onedrive links to download:
- [Raw data](https://cunicz-my.sharepoint.com/personal/22056127_cuni_cz/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F22056127%5Fcuni%5Fcz%2FDocuments%2FBakalaris%2Ddata%2Fraw%5Fdata&view=0)
- [Sorted data](https://cunicz-my.sharepoint.com/:f:/g/personal/22056127_cuni_cz/EnKfBOcrdGhCvrBcEmO8AlQB4PW03dqRuabq66f2ZQFw3Q?e=Xz6snE)

## Number of samples per genre and drum type:

||clap|crash|cymbal|hat|ride|kick|snare|tom|other|**Total**|
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|DnB|16|3|20|95|19|107|90|3|85|438|
|Dubstep|17|14|0|32|15|44|43|21|5|191|
|EDM|95|42|0|135|46|516|252|24|46|1156|
|Techno|32|0|20|121|0|119|92|14|157|555|
|Trap|71|33|8|100|35|154|117|14|134|666|
|House|151|23|30|250|26|277|231|23|123|1134|
|Trance|23|36|0|54|0|47|32|5|15|212|
|Hip-Hop|56|12|21|97|16|72|89|16|57|436|
|Funk|0|4|0|18|5|2|15|2|0|46|
|Lofi|19|2|0|41|0|26|30|0|62|180|
|Unclassified|623|309|177|1757|225|2681|2069|790|2336|10967|
|**Total**|1103|478|276|2700|387|4045|3060|912|3020|15981|