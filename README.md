# Generating Drum Kit Samples using Machine Learning
## Patrik Backo - Bachelor Thesis, 2024

In this thesis, we designed an interactive generative tool based on the Variational Autoencoder 
(VAE) to synthesise new and interesting drum one-shot samples for electronic music production. We 
researched audio representations used in audio generation tasks and selected two that suited our 
settings the best. Furthermore, we created our own dataset of almost 16,000 freely available 
samples organised into 9 drum categories. Through a series of experiments, we were able to achieve 
a model that was reconstructing and generating quality samples; however, they contained a specific 
noise artefact we could not get rid of. Based on the results of PCA and convex combinations
methods we found out that the latent space has “meaningful” properties.

## Dataset
Dataset is available on:
- [Raw data](https://cunicz-my.sharepoint.com/:f:/g/personal/22056127_cuni_cz/ElrNvHbS04FNt7k4K-4qZ90BecH-AweBeUD-4xn8GA3gww?e=kWVivM)
- [Sorted data](https://cunicz-my.sharepoint.com/:f:/g/personal/22056127_cuni_cz/EnKfBOcrdGhCvrBcEmO8AlQBeQq-iCx2xQ4FyXJAAz9ttA?e=18YsnK)

Raw data are downloaded sample libraries from the internet. Sorted data are the the data used for traning and evaluation. They were manually and programmatically cleaned from files that were not drum one-shot samples and then they were sorted into 9 categories by the type of drum sound they represent and 11 categories by the electronic music genre. (if the links do not work, please contact me)


## Instalation guide for packages necessary for the project


### Create a new conda environment
```conda create -n myenv -c conda-forge python=3.11 ```

### Activate the environment
```conda activate myenv```


### Install the packages
```conda install -c conda-forge torch librosa matplotlib scikit-learn pysoundfile scipy numpy python-sounddevice```


## Training and evaluation scripts

- **Training script**: src/VAE/training/train.py
- **Evaluation script**: src/VAE/evaluation/eval.py

All of the arguments for these scripts are documneted in the scripts themselves.



## Results attachment

The results of the experiments are available [here](https://drive.google.com/file/d/1kEZowmThN2kIpij7qlYHgufzb3iGGY8-/view?usp=sharing). There are the results for the most successful models for every major step of the research (for some of the best experminets, there is also a serialized model attached).

The best model and its results is located [here](https://drive.google.com/file/d/1E4A_e2W_eipvhmtvz-NqBrIWlktPc8qo/view?usp=sharing). It is not included in the previous link, because it was trained after the thesis was submitted.

(if you have questions or need more information, please contact me on [LinkedIn](https://www.linkedin.com/in/patrik-backo-5a9534275/?originalSubdomain=cz))
