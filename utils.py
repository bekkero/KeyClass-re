import pandas as pd
from typing import Union
from typing import List
from typing import Optional
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import trange
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import LFAnalysis
import nltk
from sentence_transformers import SentenceTransformer
import sentence_transformers
from sklearn.feature_extraction.text import CountVectorizer
from nltk import download, pos_tag, corpus
from scipy.spatial import distance
from snorkel.labeling.model.label_model import LabelModel
import torch
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
import joblib
import re
from sklearn.model_selection import train_test_split

#We define a function icd9_to_category that maps ICD-9 codes to one of 19 general diagnostic categories.
#The function takes an ICD-9 code as input, converts it to an integer if possible, and then maps it to a diagnostic category based on a predefined mapping of category ranges.
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

def load_mimic_data() :
    """
    Loads and processes medical data from the MIMIC-III database.

    Returns:
    - train: Pandas dataframe containing the training set
    - test: Pandas dataframe containing the test set
    """
    root_dir = "C:\\Users\\joshb\\PycharmProjects\\KeyClass-re\\scripts\\data\\mimic\\"

    icd_df = pd.concat([
        pd.read_csv(root_dir + 'DIAGNOSES_ICD.csv', usecols=['HADM_ID', 'ICD9_CODE']).dropna(),
        pd.read_csv(root_dir + 'PROCEDURES_ICD.csv', usecols=['HADM_ID', 'ICD9_CODE']).dropna(),
    ], ignore_index=True)

    # Concatenate the two dataframes on HADM_ID
    icd_df['ICD9_CODE'] = icd_df['ICD9_CODE'].astype(str).apply(icd9_to_category)

    # Read NOTEEVENTS.csv and filter on Discharge summary
    notes_df = pd.read_csv(root_dir + 'NOTEEVENTS.csv', usecols=['HADM_ID', 'CATEGORY', 'TEXT'])
    discharge_df = notes_df[notes_df['CATEGORY'] == 'Discharge summary']

    # Join the resulting dataframes on HADM_ID
    result_df = pd.merge(icd_df, discharge_df, on='HADM_ID', how='inner')
    # result_df = result_df.drop_duplicates(subset=['HADM_ID'], keep='first')
    result = result_df.drop_duplicates()

    target_categories = [0, 1, 2, 3]
    result = result[result['ICD9_CODE'].isin(target_categories)]
    sample_size = min(result['ICD9_CODE'].value_counts())
    sample_size = 100
    balanced_df = pd.DataFrame(columns=result.columns)
    for cat in target_categories:
        cat_df = result_df[result_df['ICD9_CODE'] == cat]
        cat_sample = cat_df.sample(sample_size, random_state=42)
        balanced_df = pd.concat([balanced_df, cat_sample])
    result = balanced_df
    #result = result.sample(80000)
    # Split the dataset into train and test sets (70/30 proportion)
    result.loc[:, 'TEXT'] = result['TEXT'].replace('\n', '', regex=True)
    train, test = train_test_split(result, test_size=0.3, random_state=42)
    return train, test

def mean_pooling(model_output, attention_mask):
    """
    Applies mean pooling to the output of a transformer-based language model.

    Arguments:
    - model_output: a tuple containing the output of a transformer-based language model
    - attention_mask: a tensor indicating which input tokens are padding tokens

    Returns:
    - mean_token_embeddings: a tensor containing the mean-pooled token embeddings
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]

    # Calculate the expanded input mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()

    # Compute the sum of token embeddings along with the input mask
    sum_token_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

    # Compute the sum of the input mask
    sum_input_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)

    # Compute the mean of the token embeddings
    mean_token_embeddings = sum_token_embeddings / sum_input_mask

    return mean_token_embeddings

def _text_length(text: Union[List[int], List[List[int]]]):
    """
    A helper function that calculates the length of the input text. The function
    can accept either a list of integers (representing a single input text) or a
    list of lists of integers (representing multiple input texts).

    Args:
    - text: A Union of a list of integers or a list of lists of integers.

    Returns:
    - An integer representing the length of the input text.
    """
    if isinstance(text, dict):  # {key: value} case
        return len(next(iter(text.values())))
    elif not hasattr(text, '__len__'):  # Object has no len() method
        return 1
    elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
        return len(text)
    else:
        return sum(len(t) for t in text)  # Sum of length of individual strings

def clean_text(sentences: Union[str, List[str]]):
    """
    Cleans a single sentence or a list of sentences by performing various preprocessing steps.

    Args:
    - sentences: A string or list of strings to be cleaned.

    Returns:
    - A list of cleaned strings.
    """

    def _clean_single_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'<.*?>|[\.`\',;\?\*\[\]\(\)-:_]*|[0-9]*', '', text)
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text

    if isinstance(sentences, str):
        sentences = [sentences]

    cleaned_sentences = [_clean_single_text(text) for text in sentences]

    return cleaned_sentences


nltk.download('stopwords')

def get_vocabulary(text_corpus, max_df=1.0, min_df=0.01, ngram_range=(1, 1)):
    """
    Returns the vocabulary and word indicator matrix for a given text corpus.

    Args:
    - text_corpus: List of strings representing the text corpus.
    - max_df: Float between 0 and 1.0, representing the maximum document frequency (default 1.0).
    - min_df: Float between 0 and 1.0, representing the minimum document frequency (default 0.01).
    - ngram_range: Tuple of integers representing the n-gram range to use (default (1,1)).

    Returns:
    - word_indicator_matrix: Numpy array of shape (n, m) where n is the number of documents and m is the number of words in the vocabulary.
    - vocabulary: Numpy array of strings representing the vocabulary.
    """
    vectorizer = CountVectorizer(max_df=max_df,
                                 min_df=min_df,
                                 strip_accents='unicode',
                                 stop_words=corpus.stopwords.words('english'),
                                 ngram_range=ngram_range)

    word_indicator_matrix = vectorizer.fit_transform(text_corpus).toarray()
    vocabulary = np.asarray(vectorizer.get_feature_names())  # Vocabulary

    return word_indicator_matrix, vocabulary

def assign_categories_to_keywords(vocabulary,
                                  vocabulary_embeddings,
                                  label_embeddings,
                                  word_indicator_matrix,
                                  cutoff=None,
                                  topk=None,
                                  min_topk=True):

    """
    Assigns categories to keywords based on cosine similarity between their embeddings.

    Args:
    - vocabulary: Numpy array of shape (num_keywords,) containing the keywords to assign categories to.
    - vocabulary_embeddings: Numpy array of shape (num_keywords, embedding_dim) containing the embeddings of the keywords.
    - label_embeddings: Numpy array of shape (num_categories, embedding_dim) containing the embeddings of the categories.
    - word_indicator_matrix: Numpy array of shape (num_documents, num_keywords) containing binary indicators for the keywords present in each document.
    - cutoff: A float specifying the maximum cosine distance between a keyword's embedding and its closest category's embedding for it to be assigned a category.
    - topk: An integer specifying the number of top categories to consider for each keyword.
    - min_topk: A boolean indicating whether to use the minimum of the topk values and the number of categories with the fewest keywords (True) or the raw topk value (False).

    Returns:
    - keywords: Numpy array of shape (num_assigned_keywords,) containing the keywords that were assigned categories.
    - assigned_category: Numpy array of shape (num_assigned_keywords,) containing the indices of the categories that the keywords were assigned to.
    - word_indicator_matrix: Numpy array of shape (num_documents, num_assigned_keywords) containing binary indicators for the keywords that were assigned categories.
    """

    assert (cutoff is None) or (topk is None)

    distances = distance.cdist(vocabulary_embeddings, label_embeddings, 'cosine')

    dist_to_closest_cat = np.min(distances, axis=1)
    assigned_category = np.argmin(distances, axis=1)

    if cutoff is not None:
        mask = dist_to_closest_cat <= cutoff
    elif topk is not None:
        unique_categories = np.unique(assigned_category)
        mask = np.zeros(len(dist_to_closest_cat), dtype=bool)
        _, counts = np.unique(assigned_category, return_counts=True)

        if min_topk:
            topk = min(topk, np.min(counts))

        for unique_category in unique_categories:
            category_indices = np.where(assigned_category == unique_category)[0]
            category_distances = dist_to_closest_cat[category_indices]
            sorted_indices = np.argsort(category_distances)[:topk]
            mask[category_indices[sorted_indices]] = True
    else:
        mask = np.ones(len(dist_to_closest_cat), dtype=bool)

    keywords = vocabulary[mask]
    assigned_category = assigned_category[mask]
    word_indicator_matrix = word_indicator_matrix[:, np.where(mask)[0]]
    return keywords, assigned_category, word_indicator_matrix


def create_label_matrix(word_indicator_matrix, keywords, assigned_category):
    """
    Creates a label matrix from a word indicator matrix, using specified keywords and assigned categories.

    Args:
    - word_indicator_matrix: Numpy array of shape (n_samples, n_features) containing binary indicators of word occurrence.
    - keywords: List of keywords used to construct the word indicator matrix.
    - assigned_category: List of integers representing the assigned category for each keyword.

    Returns:
    - label_matrix: Pandas dataframe of shape (n_samples, n_categories) containing assigned categories for each sample.
    """

    word_indicator_matrix = np.where(word_indicator_matrix == 0, -1, 0)
    for i, category in enumerate(assigned_category):
        word_indicator_matrix[:, i] = np.where(word_indicator_matrix[:, i] != -1, category, -1)
    return pd.DataFrame(word_indicator_matrix, columns=keywords)

def print_pretty(msg, metrics):
    """Logging function.

    Parameters
    ----------
    metrics: Union[List, Dict]
        The metrics to log and save in a file
    """
    if isinstance(metrics, list):
        assert len(metrics) == 3, "Metrics must be of length 3!"
        columns = ["Accuracy", "Precision", "Recall"]
        data = [metrics[0], metrics[1], metrics[2]]
    elif isinstance(metrics, np.ndarray):
        assert len(metrics) == 3, "Metrics must be of length 3!"
        columns = ["Accuracy (mean, std)", "Precision (mean, std)", "Recall (mean, std)"]
        data = [f"{np.round(metrics[0][0], 4)}, {np.round(metrics[0][1], 4)}",
                f"{np.round(metrics[1][0], 4)}, {np.round(metrics[1][1], 4)}",
                f"{np.round(metrics[2][0], 4)}, {np.round(metrics[2][1], 4)}"]
    else:
        columns = list(metrics.keys())
        data = list(metrics.values())

    table = pd.DataFrame([data], columns=columns)
    print(msg + ":")
    print(table.to_string(index=False))

def compute_metrics_bootstrap(y_preds: np.array,
                              y_true: np.array,
                              average: str = 'weighted',
                              n_bootstrap: int = 100,
                              n_jobs: int = 10) -> np.array:
    """
    Compute bootstrapped confidence intervals (CIs) around metrics of interest.

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
        Number of bootstrap samples to compute CI.

    n_jobs: int
        Number of jobs to run in parallel.
    """

    bootstrap_indices = [
        np.random.choice(a=len(y_true), size=len(y_true))
        for _ in range(n_bootstrap)
    ]

    output = joblib.Parallel(n_jobs=n_jobs, verbose=1)(
        joblib.delayed(compute_metrics)(y_preds[bootstrap_inds], y_true[bootstrap_inds])
        for bootstrap_inds in bootstrap_indices
    )

    output = np.array(output)
    means = np.mean(output, axis=0)
    stds = np.std(output, axis=0)

    return np.stack([means, stds], axis=1)

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
        f1_score(y_true,y_preds,average=average),
        precision_score(y_true, y_preds, average=average),
        recall_score(y_true, y_preds, average=average)
    ]

def get_balanced_data_mask(proba_preds: np.array,
                           max_num: int = 7000,
                           class_balance: Optional[np.array] = None) -> np.array:
    """
    Utility function to keep only the most confident predictions, while maintaining class balance

    Parameters
    ----------
    proba_preds: Probabilistic labels of data points
    max_num: Maximum number of data points per class
    class_balance: Prevalence of each class
    """
    num_classes = proba_preds.shape[1]

    if class_balance is None:  # Assume all classes are equally likely
        class_balance = np.ones(num_classes) / num_classes

    assert abs(np.sum(class_balance) - 1) < 1e-3, "Class balance must be a probability, and hence sum to 1"
    assert len(class_balance) == num_classes, f"Only {num_classes} classes in the data"

    # Get integer of max number of elements per class
    class_max_inds = [int(max_num * c) for c in class_balance]
    train_idxs = np.array([], dtype=int)

    for i in range(num_classes):
        sorted_idxs = np.argsort(proba_preds[:, i])[::-1]  # gets highest probas for class
        sorted_idxs = sorted_idxs[:class_max_inds[i]]
        print(f'Confidence of least confident data point of class {i}: {proba_preds[sorted_idxs[-1], i]}')
        train_idxs = np.union1d(train_idxs, sorted_idxs)

    mask = np.zeros(len(proba_preds), dtype=bool)
    mask[train_idxs] = True
    return mask

def load_data(proba_preds,max_num,train_notes_embeddings,test_notes_embeddings,y_train,y_test):
    """
    Loads and processes training and testing data for a machine learning model.

    Args:
    - proba_preds: Numpy array of predicted probabilities from a label model
    - max_num: Integer specifying the maximum number of data points to keep
    - train_notes_embeddings: Numpy array of training note embeddings
    - test_notes_embeddings: Numpy array of testing note embeddings
    - y_train: Numpy array of ground truth labels for the training data
    - y_test: Numpy array of ground truth labels for the testing data

    Returns:
    - X_train_embed_masked: Numpy array of masked training note embeddings
    - y_train_lm_masked: Numpy array of masked label model predictions for the training data
    - y_train_masked: Numpy array of masked ground truth labels for the training data
    - X_test_embed: Numpy array of testing note embeddings
    - y_test: Numpy array of ground truth labels for the testing data
    - sample_weights_masked: Numpy array of sample weights for noise aware loss
    - proba_preds_masked: Numpy array of masked predicted probabilities from the label model
    """
    y_train_lm = np.argmax(proba_preds, axis=1)
    sample_weights = np.max(proba_preds,axis=1)  # Sample weights for noise aware loss

    # Keep only very confident predictions
    mask =  get_balanced_data_mask(proba_preds, max_num=max_num, class_balance=None)

    # Load training and testing data
        # We have already encode the dataset, so we'll just load the embeddings
    X_train_embed = train_notes_embeddings
    X_test_embed = test_notes_embeddings


    print('\n==== Data statistics ====')
    print(
        f'Size of training data: {X_train_embed.shape}, testing data: {X_test_embed.shape}'
    )
    print(f'Size of testing labels: {y_test.shape}')
    print(f'Size of training labels: {y_train.shape}')
    print(
            f'Training class distribution (ground truth): {np.unique(y_train, return_counts=True)[1]/len(y_train)}'
    )
    print(
        f'Training class distribution (label model predictions): {np.unique(y_train_lm, return_counts=True)[1]/len(y_train_lm)}'
    )
    print(
        '\nKeyClass only trains on the most confidently labeled data points! Applying mask...'
    )
    print('\n==== Data statistics (after applying mask) ====')
    y_train_masked = y_train[mask]
    y_train_lm_masked = y_train_lm[mask]
    X_train_embed_masked = X_train_embed[mask]
    sample_weights_masked = sample_weights[mask]
    proba_preds_masked = proba_preds[mask]

    print(
            f'Training class distribution (ground truth): {np.unique(y_train_masked, return_counts=True)[1]/len(y_train_masked)}'
        )
    print(
        f'Training class distribution (label model predictions): {np.unique(y_train_lm_masked, return_counts=True)[1]/len(y_train_lm_masked)}'
    )

    return X_train_embed_masked, y_train_lm_masked, y_train_masked, X_test_embed, y_test, sample_weights_masked, proba_preds_masked