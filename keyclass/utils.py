# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
from os.path import join, exists
import re
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score
from datetime import datetime
import torch
from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper
import pandas as pd
from sklearn.model_selection import train_test_split


def log(metrics: Union[List, Dict], filename: str, results_dir: str,
        split: str):
    """Logging function
        
        Parameters
        ----------
        metrics: Union[List, Dict]
            The metrics to log and save in a file
        filename: str
            Name of the file
        results_dir: str
            Path to results directory
        split: str
            Train/test split
    """
    if isinstance(metrics, list):
        assert len(metrics) == 3, "Metrics must be of length 3!"
        results = dict()
        results['Accuracy'] = metrics[0]
        results['Precision'] = metrics[1]
        results['Recall'] = metrics[2]
    elif isinstance(metrics, np.ndarray):
        assert len(metrics) == 3, "Metrics must be of length 3!"
        results = dict()
        results['Accuracy (mean, std)'] = metrics[0].tolist()
        results['Precision (mean, std)'] = metrics[1].tolist()
        results['Recall (mean, std)'] = metrics[2].tolist()
    else:
        results = metrics

    filename_complete = join(
        results_dir,
        f'{split}_{filename}_{datetime.now().strftime("%d-%b-%Y-%H_%M_%S")}.txt'
    )
    print(f'Saving results in {filename_complete}...')

    with open(filename_complete, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results))


def compute_metrics(y_preds: np.array,
                    y_true: np.array,
                    average: str = 'weighted'):
    """Compute accuracy, recall and precision

        Parameters
        ----------
        y_preds: np.array
            Predictions
        
        y_true: np.array
            Ground truth labels
        
        average: str
            This parameter is required for multiclass/multilabel targets. If None, 
            the scores for each class are returned. Otherwise, this determines the 
            type of averaging performed on the data.
    """
    return [
        np.mean(y_preds == y_true),
        precision_score(y_true, y_preds, average=average),
        recall_score(y_true, y_preds, average=average)
    ]


def compute_metrics_bootstrap(y_preds: np.array,
                              y_true: np.array,
                              average: str = 'weighted',
                              n_bootstrap: int = 100,
                              n_jobs: int = 10):
    """Compute bootstrapped confidence intervals (CIs) around metrics of interest. 

        Parameters
        ----------
        y_preds: np.array
            Predictions
        
        y_true: np.array
            Ground truth labels
        
        average: str
            This parameter is required for multiclass/multilabel targets. If None, 
            the scores for each class are returned. Otherwise, this determines the 
            type of averaging performed on the data.

        n_bootstrap: int
            Number of boostrap samples to compute CI. 

        n_jobs: int
            Number of jobs to run in parallel. 
    """
    output_ = joblib.Parallel(n_jobs=n_jobs, verbose=1)(
        joblib.delayed(compute_metrics)
        (y_preds[boostrap_inds], y_true[boostrap_inds]) \
        for boostrap_inds in [ \
            np.random.choice(a=len(y_true), size=len(y_true)) for k in range(n_bootstrap)])
    output_ = np.array(output_)
    means = np.mean(output_, axis=0)
    stds = np.std(output_, axis=0)
    return np.stack([means, stds], axis=1)


def get_balanced_data_mask(proba_preds: np.array,
                           max_num: int = 7000,
                           class_balance: Optional[np.array] = None):
    """Utility function to keep only the most confident predictions, while maintaining class balance

        Parameters
        ---------- 
        proba_preds: Probabilistic labels of data points
        max_num: Maximum number of data points per class
        class_balance: Prevalence of each class

    """
    if class_balance is None:  # Assume all classes are equally likely
        class_balance = np.ones(proba_preds.shape[1]) / proba_preds.shape[1]

    assert np.sum(
        class_balance
    ) - 1 < 1e-3, "Class balance must be a probability, and hence sum to 1"
    assert len(class_balance) == proba_preds.shape[
        1], f"Only {proba_preds.shape[1]} classes in the data"

    # Get integer of max number of elements per class
    class_max_inds = [int(max_num * c) for c in class_balance]
    train_idxs = np.array([], dtype=int)

    for i in range(proba_preds.shape[1]):
        sorted_idxs = np.argsort(
            proba_preds[:, i])[::-1]  # gets highest probas for class
        sorted_idxs = sorted_idxs[:class_max_inds[i]]
        print(
            f'Confidence of least confident data point of class {i}: {proba_preds[sorted_idxs[-1], i]}'
        )
        train_idxs = np.union1d(train_idxs, sorted_idxs)

    mask = np.zeros(len(proba_preds), dtype=bool)
    mask[train_idxs] = True
    return mask


dx_cache = {}
sg_cache = {}


def clean(text):
    text = text.lower()
    text = re.sub(r'<.*?>|[\.`\',;\?\*\[\]\(\)-:_]*|[0-9]*', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    #text = text.replace("\n", "")

    return text


import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def select_most_common_words(descriptions) :
    words = nltk.word_tokenize(descriptions.lower())
    # Remove stop words and count the frequency of each remaining word
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)

    # Store the top 10 most frequently occurring words for the current category
    top_words = [word for word, count in word_counts.most_common(75)]
    return ' '.join(top_words)

def get_diag_category(category_num):
    root_dir = "//scripts/data/mimic/"
    # Create a set to store unique descriptions
    # descriptions = set()
    descriptions = ""

    # Check if the DX cache is empty and load it if necessary
    if not dx_cache:
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words = set(stopwords.words('english'))
        #with open(root_dir + "D_ICD_DIAGNOSES.csv", "r") as dx_file:
        with open(root_dir + "CMS32_DESC_LONG_DX.txt", "r") as dx_file:
            next(dx_file)
            for line in dx_file:
                #fields = line.split(",")
                #fields = line.split(' ',1)
                #code_str = fields[1].strip('"')
                code_str, description = line.split(' ', 1)
                code = icd9_to_category(code_str)
                #description = fields[3].strip('"')
                description = description.strip("'")
                if code not in dx_cache:
                    dx_cache[code] = clean(description + " ")
                else:
                    dx_cache[code] += clean(description + " ")

                #descriptions = dx_cache[code]
                #dx_cache[code] = select_most_common_words(' '.join((list(set(descriptions.split())))))

    # Check if the SG cache is empty and load it if necessary
    if not sg_cache:
        #with open(root_dir + "D_ICD_PROCEDURES.csv", "r") as sg_file:
        with open(root_dir + "CMS32_DESC_LONG_SG.txt", "r") as sg_file:
            next(sg_file)
            for line in sg_file:
                #fields = line.split(",")
                #code_str = fields[1].strip('"')
                code_str, description = line.split(' ', 1)
                code = icd9_to_category(code_str)
                #description = fields[3].strip('"')
                description = description.strip("'")
                if code not in sg_cache:
                    sg_cache[code] = clean(description + " ")
                else:
                    sg_cache[code] += clean(description + " ")
                #descriptions = sg_cache[code]
                #sg_cache[code] = select_most_common_words(' '.join((list(set(descriptions.split())))))


    if category_num in sg_cache:
        descriptions = sg_cache[category_num] + " "
    if category_num in dx_cache:
        descriptions += dx_cache[category_num] + " "


    #return ' '.join((list(set(descriptions.split()))))
    return select_most_common_words(' '.join((list(set(descriptions.split())))))


def icd9_to_category(icd9_code):
    # Define the mapping of ICD-9 codes to general diagnostic categories
    category_mapping = {
        0: (0, 139),
        1: (140, 239),
        2: (240, 279),
        3: (280, 289),
        4: (290, 319),
        5: (320, 389),
        6: (390, 459),
        7: (460, 519),
        8: (520, 579),
        9: (580, 629),
        10: (630, 679),
        11: (680, 709),
        12: (710, 739),
        13: (740, 759),
        14: (760, 779),
        15: (780, 799),
        16: (800, 999),
        17: (1000, 2000),  # For E and V codes (e.g., E000-E999, V01-V91)
        18: (2001, 3000),  # For U codes (e.g., U000-U999)
    }
    # Convert the ICD-9 code to an integer, if possible
    try:
        # first_three_digits = int(icd9_code[:3])
        code_num = int(icd9_code[:3])
    except ValueError:
        # Handle ICD-9 codes starting with 'E', 'V', or 'U'
        if icd9_code.startswith('E'):
            code_num = 1000
        elif icd9_code.startswith('V'):
            code_num = 1000
        elif icd9_code.startswith('U'):
            code_num = 2001
        else:
            raise ValueError(f"Invalid ICD-9 code: {icd9_code}")
    # Map the ICD-9 code to one of the 19 general diagnostic categories

    for category, (start, end) in category_mapping.items():
        if start <= code_num <= end:
            return category

    # raise ValueError(f"ICD-9 code '{icd9_code}' is out of the range of general diagnostic categories.")
    return -1


def load_mimic_data():
    root_dir = "//scripts/data/mimic/"

    # Read DIAGNOSES_ICD.csv and PROCEDURES_ICD.csv
    #diag_df = pd.read_csv(root_dir + 'DIAGNOSES_ICD.csv', usecols=['HADM_ID', 'ICD9_CODE'])
    #diag_df = diag_df.dropna()
    #diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str)
    #diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].apply(icd9_to_category)
    #diag_df = diag_df.drop_duplicates()

    #proc_df = pd.read_csv(root_dir + 'PROCEDURES_ICD.csv', usecols=['HADM_ID', 'ICD9_CODE'])
    #proc_df = proc_df.dropna()
    #proc_df['ICD9_CODE'] = proc_df['ICD9_CODE'].astype(str)
    #proc_df['ICD9_CODE'] = proc_df['ICD9_CODE'].apply(icd9_to_category)
    #proc_df = proc_df.drop_duplicates()

    icd_df = pd.concat([
        pd.read_csv(root_dir + 'DIAGNOSES_ICD.csv', usecols=['HADM_ID', 'ICD9_CODE']).dropna(),
        pd.read_csv(root_dir + 'PROCEDURES_ICD.csv', usecols=['HADM_ID', 'ICD9_CODE']).dropna(),
    ], ignore_index=True)

    # Concatenate the two dataframes on HADM_ID
    #icd_df = pd.concat([diag_df, proc_df], ignore_index=True)
    #icd_df = icd_df.drop_duplicates()
    icd_df['ICD9_CODE'] = icd_df['ICD9_CODE'].astype(str).apply(icd9_to_category)

    # Read NOTEEVENTS.csv and filter on Discharge summary
    notes_df = pd.read_csv(root_dir + 'NOTEEVENTS.csv', usecols=['HADM_ID','CATEGORY', 'TEXT'])
    discharge_df = notes_df[notes_df['CATEGORY'] == 'Discharge summary']

    # Join the resulting dataframes on HADM_ID
    result_df = pd.merge(icd_df, discharge_df, on='HADM_ID', how='inner')
    #result_df = result_df.drop_duplicates(subset=['HADM_ID'], keep='first')
    result = result_df.drop_duplicates()

    target_categories = [0, 1, 2, 3, 4, 5]
    result = result[result['ICD9_CODE'].isin(target_categories)]
    sample_size = min(result['ICD9_CODE'].value_counts())
    balanced_df = pd.DataFrame(columns=result.columns)
    for cat in target_categories:
        cat_df = result_df[result_df['ICD9_CODE'] == cat]
        cat_sample = cat_df.sample(sample_size, random_state=42)
        balanced_df = pd.concat([balanced_df, cat_sample])
    result = balanced_df
    #result = result.sample(75000)

    # Split the dataset into train and test sets (70/30 proportion)
    #result['TEXT'] = result['TEXT'].replace('\n', '', regex=True)
    result.loc[:, 'TEXT'] = result['TEXT'].replace('\n', '', regex=True)
    train, test = train_test_split(result, test_size=0.3, random_state=42)

    train[['TEXT']].to_csv(root_dir + 'train.txt', index=False, header=False)
    train[['ICD9_CODE']].to_csv(root_dir + 'train_labels.txt', index=False, header=False)

    test[['TEXT']].to_csv(root_dir + 'test.txt', index=False, header=False)
    test[['ICD9_CODE']].to_csv(root_dir + 'test_labels.txt', index=False, header=False)


def clean_text(sentences: Union[str, List[str]]):
    """Utility function to clean sentences
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    for i, text in enumerate(sentences):
        text = text.lower()
        text = re.sub(r'<.*?>|[\.`\',;\?\*\[\]\(\)-:_]*|[0-9]*', '', text)
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        sentences[i] = text

    # return [string for sublist in sentences for string in sublist]
    return sentences


def fetch_data(dataset='imdb', path='~/', split='train'):
    """Fetches a dataset by its name

	    Parameters
	    ---------- 
	    dataset: str
	        List of text to be encoded. 

	    path: str
	        Path to the stored data. 

	    split: str
	        Whether to fetch the train or test dataset. Options are one of 'train' or 'test'. 
    """
    # _dataset_names = ['agnews', 'amazon', 'dbpedia', 'imdb', 'mimic']
    # if dataset not in _dataset_names:
    #    raise ValueError(f'Dataset must be one of {_dataset_names}, but received {dataset}.')
    # if split not in ['train', 'test']:
    #    raise ValueError(f'split must be one of \'train\' or \'test\', but received {split}.')
    if dataset == 'mimic' and not exists(f"{join(path, dataset, split)}.txt"):
        load_mimic_data()

    if not exists(f"{join(path, dataset, split)}.txt"):
        raise ValueError(
            f'File {split}.txt does not exists in {join(path, dataset)}')

    text = open(f'{join(path, dataset, split)}.txt', encoding='utf-8').readlines()

    if dataset == 'mimic':
        # text = [clean_text(line) for line in text]
        text = [string for line in text for string in clean_text(line)]

    return text


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def _text_length(text: Union[List[int], List[List[int]]]):
    """
    Help function to get the length for the input text. Text can be either
    a list of ints (which means a single text as input), or a tuple of list of ints
    (representing several text inputs to the model).

    Adapted from https://github.com/UKPLab/sentence-transformers/blob/40af04ed70e16408f466faaa5243bee6f476b96e/sentence_transformers/SentenceTransformer.py#L548
    """

    if isinstance(text, dict):  # {key: value} case
        return len(next(iter(text.values())))
    elif not hasattr(text, '__len__'):  # Object has no len() method
        return 1
    elif len(text) == 0 or isinstance(text[0],
                                      int):  # Empty string or list of ints
        return len(text)
    else:
        return sum([len(t)
                    for t in text])  # Sum of length of individual strings


class Parser:

    def __init__(
            self,
            config_file_path='../config_files/default_config.yml',
            default_config_file_path='../config_files/default_config.yml'):
        """Class to read and parse the config.yml file
		"""
        self.config_file_path = config_file_path
        with open(default_config_file_path, 'rb') as f:
            self.default_config = load(f, Loader=Loader)

    def parse(self):
        with open(self.config_file_path, 'rb') as f:
            self.config = load(f, Loader=Loader)

        for key, value in self.default_config.items():
            if ('target' not in key) and ((key not in list(self.config.keys()))
                                          or (self.config[key] is None)):
                self.config[key] = self.default_config[key]
                print(
                    f'Setting the value of {key} to {self.default_config[key]}!'
                )

        target_present = False
        for key in self.config.keys():
            if 'target' in key:
                target_present = True
                break
        if not target_present: raise ValueError("Target must be present.")
        self.save_config()
        return self.config

    def save_config(self):
        with open(self.config_file_path, 'w') as f:
            dump(self.config, f)
