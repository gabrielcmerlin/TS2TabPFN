import os
import numpy as np
import pandas as pd
from aeon.datasets import load_from_ts_file
from sklearn.utils import check_random_state

def load_experiment_data(problem_path: str,dataset: str,resample_id: int,predefined_resample: bool):
    
    if resample_id is not None and predefined_resample:
        resample_str = "" if resample_id is None else str(resample_id)

        X_train, y_train = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TRAIN.ts"
        )
        X_test, y_test = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TEST.ts"
        )

        resample_data = False
    else:
        X_train, y_train = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}_TRAIN.ts"
        )
        X_test, y_test = load_from_ts_file(
            f"{problem_path}/{dataset}/{dataset}_TEST.ts"
        )

        resample_data = True if resample_id != 0 else False

    return X_train, y_train, X_test, y_test, resample_data

def resample_data(X_train, y_train, X_test, y_test, random_state=None):

    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)
    indices = np.arange(len(all_data), dtype=int)
    rng.shuffle(indices)

    train_indices = indices[: len(X_train)]
    test_indices = indices[len(X_train) :]

    # split the shuffled data into train and test
    X_train = (
        all_data[train_indices] if is_array else [all_data[i] for i in train_indices]
    )
    y_train = all_labels[train_indices]
    X_test = all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
    y_test = all_labels[test_indices]

    return X_train, y_train, X_test, y_test

def stratified_resample_data(X_train, y_train, X_test, y_test, random_state=None):
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    if is_array:
        shape = list(X_train.shape)
        shape[0] = 0

    X_train = np.zeros(shape) if is_array else []
    y_train = np.zeros(0)
    X_test = np.zeros(shape) if is_array else []
    y_test = np.zeros(0)

    # for each class
    for label_index in range(len(unique_train)):
        # get the indices of all instances with this class label and shuffle them
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        rng.shuffle(indices)

        train_indices = indices[: counts_train[label_index]]
        test_indices = indices[counts_train[label_index] :]

        # extract data from corresponding indices
        train_cases = (
            all_data[train_indices]
            if is_array
            else [all_data[i] for i in train_indices]
        )
        train_labels = all_labels[train_indices]
        test_cases = (
            all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
        )
        test_labels = all_labels[test_indices]

        # concat onto current data from previous loop iterations
        X_train = (
            np.concatenate([X_train, train_cases], axis=0)
            if is_array
            else X_train + train_cases
        )
        y_train = np.concatenate([y_train, train_labels], axis=None)
        X_test = (
            np.concatenate([X_test, test_cases], axis=0)
            if is_array
            else X_test + test_cases
        )
        y_test = np.concatenate([y_test, test_labels], axis=None)

    return X_train, y_train, X_test, y_test

def get_data_reg(data_path, dataset_name, resample_id, predefined_resample=False):
    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        data_path, dataset_name, resample_id, predefined_resample
    )

    if resample:
        X_train, y_train, X_test, y_test = resample_data(
            X_train, y_train, X_test, y_test, random_state=resample_id
        )

    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    return X_train, y_train, X_test, y_test

def get_data_clf(data_path, dataset_name, resample_id, predefined_resample=False):
    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        data_path, dataset_name, resample_id, predefined_resample
    )

    if resample:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train, y_train, X_test, y_test, random_state=resample_id
        )
    
    return X_train, y_train, X_test, y_test

def get_data_extracted_reg(feature_cache_path,dataset_name,resample_id):

    X_full = np.load(os.path.join(feature_cache_path, f"{dataset_name}_X.npy"))
    y_full = np.load(os.path.join(feature_cache_path, f"{dataset_name}_y.npy"))
    n_train = int(np.load(os.path.join(feature_cache_path, f"{dataset_name}_ntrain.npy"))[0])

    X_train = X_full[:n_train]
    X_test = X_full[n_train:]
    y_train = y_full[:n_train]
    y_test = y_full[n_train:]

    resample = True if resample_id != 0 else False
    if resample:
        X_train, y_train, X_test, y_test = resample_data(
            X_train,
            y_train,
            X_test,
            y_test,
            random_state=resample_id
        )

    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    return X_train, y_train, X_test, y_test

def get_data_extracted_clf(feature_cache_path,dataset_name,resample_id):

    X_full = np.load(os.path.join(feature_cache_path, f"{dataset_name}_X.npy"))
    y_full = np.load(os.path.join(feature_cache_path, f"{dataset_name}_y.npy"))
    n_train = int(np.load(os.path.join(feature_cache_path, f"{dataset_name}_ntrain.npy"))[0])

    X_train = X_full[:n_train]
    X_test = X_full[n_train:]
    y_train = y_full[:n_train]
    y_test = y_full[n_train:]

    resample = True if resample_id != 0 else False
    if resample:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train, y_train, X_test, y_test, random_state=resample_id
        )

    return X_train, y_train, X_test, y_test