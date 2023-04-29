# KeyClass-re
Reproduction experiment

KeyClass-re is an attempt to reproduce the experiment and validate claims made in the paper Gao, C., Goswami, M., Chen, J., & Dubrawski, A. (2020). Classifying Unstructured Clinical Notes via Automatic Weak Supervision. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 6257-6268. doi: 10.18653/v1/2020.emnlp-main.509

The attempt was carried out on a Window 10 machine with GPU. To set up  the environment run :

conda env create -f KeyClassWindows10.yml

There are two additinal file

KeyClassReproduction.ipynb - A jupiter notebook that has full experiment with explanations

utils.py - a set od helper functions.

MIMIC-III dataset is another dependency. The dataset is not provided part of this repo. For the experiment the below files from the datasets are required:

DIAGNOSES_ICD.csv
PROCEDURES_ICD.csv

