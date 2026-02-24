import numpy as np
import pandas as pd
from aeon.datasets import load_from_ts_file, write_to_ts_file
from sklearn.utils import check_random_state

def to_tsfresh_df(X):
    if X.ndim == 2:
        X = X[:, None, :]
    N, C, T = X.shape
    dfs = []
    for i in range(N):
        for c in range(C):
            dfs.append(
                pd.DataFrame({
                    "id": i,
                    "time": np.arange(T),
                    "value": X[i, c],
                    "kind": f"ch{c}"
                })
            )
    return pd.concat(dfs, ignore_index=True)

def adjust_ts_length(X, num_patches):
    T = X.shape[-1]
    T_new = (T // num_patches) * num_patches
    return X[..., :T_new]

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

def get_data(data_path, dataset_name, resample_id, predefined_resample=False):

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