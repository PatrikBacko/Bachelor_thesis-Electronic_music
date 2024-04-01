from pathlib import Path
import os
import shutil
import numpy as np

data_path = Path(r'C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music\data\debug')

for sample_group in data_path.iterdir():
    sample_path = sample_group / f'{sample_group.name}_samples' 
    test_path = sample_group / f'{sample_group.name}_test_samples'

    test_path.mkdir(exist_ok=True)

    sampel = np.random.choice(os.listdir(sample_path))

    shutil.copyfile(sample_path / sampel, test_path / sampel)

    