import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from aeon.transformations.collection.feature_based import Catch22
from aeon.regression.interval_based import DrCIFRegressor
from tsfresh import extract_features
from tabpfn_extensions.many_class import ManyClassClassifier
from tabpfn import TabPFNRegressor, TabPFNClassifier
from aeon.regression.convolution_based import MiniRocketRegressor
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.classification.hybrid import HIVECOTEV2
from codecarbon import EmissionsTracker
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)

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

def extract_feat(X_train, X_test, model):
    was_extracted = False

    if model == 'c22':
        catch22 = Catch22()
        X_train = catch22.fit_transform(X_train)
        X_test = catch22.transform(X_test)
        was_extracted = True

    elif model == 'multirocket_fm':
        multirocket = MultiRocket()
        multirocket.fit(X_train)

        X_train_feat = multirocket.transform(X_train)
        X_test_feat = multirocket.transform(X_test)

        X_train = np.asarray(X_train_feat)
        X_test = np.asarray(X_test_feat)
        was_extracted = True

    elif model == 'tsfresh':
        n_train = len(X_train)
        X_full = np.concatenate([X_train, X_test], axis=0)
        
        df_full = to_tsfresh_df(X_full)
        
        X_full_feat = extract_features(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=16
        )
        
        impute(X_full_feat)
        X_full_feat = X_full_feat.to_numpy()
        
        X_train = X_full_feat[:n_train]
        X_test = X_full_feat[n_train:]
        was_extracted = True

    return X_train, X_test, was_extracted

def train_test_model_reg(X_train, X_test, y_train, SEED, run, DEVICE, model_name, emissions_dir=None):
    if model_name == 'MiniRocket':
        regressor = MiniRocketRegressor(random_state=SEED+run)
    elif model_name == 'DrCIF':
        d = X_train.shape[1] 
        m = X_train.shape[2]
        n_intervals = int(4 + (np.sqrt(d) * np.sqrt(m)) / 3)
        regressor = DrCIFRegressor(
            n_estimators=500,
            n_intervals=n_intervals,
            random_state=SEED + run,
            n_jobs=-1
        )
    else:    
        regressor = TabPFNRegressor(
            random_state=SEED+run,
            ignore_pretraining_limits=True,
            device=DEVICE
        )

    tracker_train = EmissionsTracker(
        project_name=f"{model_name}_run{run+1}_train",
        output_dir=emissions_dir,
        output_file="emissions_train.csv",
        log_level="error"
    )
    
    start_train = time.time()
    tracker_train.start()
    try:
        regressor.fit(X_train, y_train)
    finally:
        tracker_train.stop()
    train_time = time.time() - start_train

    tracker_test = EmissionsTracker(
        project_name=f"{model_name}_run{run+1}_test",
        output_dir=emissions_dir,
        output_file="emissions_test.csv",
        log_level="error"
    )

    start_test = time.time()
    tracker_test.start()
    try:
        y_pred = regressor.predict(X_test)
    finally:
        tracker_test.stop()
    test_time = time.time() - start_test

    return y_pred, train_time, test_time
    
def train_test_model_clf(X_train, X_test, y_train, SEED, run, DEVICE, model_name, emissions_dir=None):
    if model_name == 'HC2':
        classifier = HIVECOTEV2(
            random_state=SEED + run,
            n_jobs=-1
        )
    else:
        estimator = TabPFNClassifier(
            random_state=SEED+run,
            ignore_pretraining_limits=True,
            device=DEVICE
        )
        classifier = ManyClassClassifier(estimator=estimator, alphabet_size=10)

    tracker_train = EmissionsTracker(
        project_name=f"{model_name}_run{run+1}_train",
        output_dir=emissions_dir,
        output_file="emissions_train.csv",
        log_level="error"
    )

    start_train = time.time()
    tracker_train.start()
    try:
        classifier.fit(X_train, y_train)
    finally:
        tracker_train.stop()
    train_time = time.time() - start_train

    tracker_test = EmissionsTracker(
        project_name=f"{model_name}_run{run+1}_test",
        output_dir=emissions_dir,
        output_file="emissions_test.csv",
        log_level="error"
    )

    start_test = time.time()
    tracker_test.start()
    try:
        y_pred = classifier.predict(X_test)
    finally:
        tracker_test.stop()
    test_time = time.time() - start_test

    return y_pred, train_time, test_time

def get_results_reg(y_test, y_pred, model_name, dataset_name, train_time, test_time, run):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    row = {
        "model": model_name,
        "dataset": dataset_name,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_time": train_time,
        "test_time": test_time,
        "run": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run+1}"
    }
    return row

def get_results_clf(y_test, y_pred, model_name, dataset_name, train_time, test_time, run):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    row = {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "train_time": train_time,
        "test_time": test_time,
        "run": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run+1}"
    }
    return row

def store_results(results, csv_path):
    df_row = pd.DataFrame([results])
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

def print_results_reg(results):
    print("\n           ===== Results =====")
    print(f"           MSE       : {results['mse']:.8f}")
    print(f"           MAE       : {results['mae']:.8f}")
    print(f"           RMSE      : {results['rmse']:.8f}")
    print(f"           R²        : {results['r2']:.8f}")
    print(f"           Train Time: {results['train_time']:.4f}s")
    print(f"           Test Time : {results['test_time']:.4f}s")

def print_results_clf(results):
    print("\n           ===== Results =====")
    print(f"           Accuracy  : {results['accuracy']:.8f}")
    print(f"           F1-score  : {results['f1']:.8f}")
    print(f"           Precision : {results['precision']:.8f}")
    print(f"           Recall    : {results['recall']:.8f}")
    print(f"           Train Time: {results['train_time']:.4f}s")
    print(f"           Test Time : {results['test_time']:.4f}s")