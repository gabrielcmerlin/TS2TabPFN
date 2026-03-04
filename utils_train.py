import os
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
from sklearn.metrics import (
    mean_squared_error,mean_absolute_error,r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)

def extract_feat(X_train, X_test, model):
    
    if model == 'c22':
        catch22 = Catch22()
        X_train = catch22.fit_transform(X_train)
        X_test = catch22.transform(X_test)

    elif model == 'multirocket_fm':
        multirocket = MultiRocket()
        multirocket.fit(X_train)

        X_train_feat = multirocket.transform(X_train)
        X_test_feat = multirocket.transform(X_test)

        X_train = np.asarray(X_train_feat)
        X_test = np.asarray(X_test_feat)

    return X_train, X_test

def train_model_reg(X_train, X_test, y_train, SEED, run, DEVICE, model_name):
    if model_name == 'MiniRocket':
        regressor = MiniRocketRegressor(
            random_state=SEED+run,
        )
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

    else:    
        regressor = TabPFNRegressor(
            random_state=SEED+run,
            ignore_pretraining_limits=True,
            device=DEVICE
        )
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

    return y_pred
    
def train_model_clf(X_train, X_test, y_train, SEED, run, DEVICE, model_name):
    estimator = TabPFNClassifier(
        random_state=SEED+run,
        ignore_pretraining_limits=True,
        device=DEVICE
    )
    classifier = ManyClassClassifier(estimator=estimator,alphabet_size=10)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return y_pred
    
def get_results_reg(y_test, y_pred, model_name, dataset_name, elapsed_time, run):
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
        "time": elapsed_time,
        "run": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run+1}"
    }

    return row

def get_results_clf(y_test, y_pred, model_name, dataset_name, elapsed_time, run):
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
        "time": elapsed_time,
        "run": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run+1}"
    }

    return row

def store_results(results, csv_path):
    df_row = pd.DataFrame([results])

    # Append no CSV
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

def print_results_reg(results):
    print("\n           ===== Results =====")
    print(f"           MSE : {results['mse']:.8f}")
    print(f"           MAE : {results['mae']:.8f}")
    print(f"           RMSE: {results['rmse']:.8f}")
    print(f"           R²  : {results['r2']:.8f}")

def print_results_clf(results):
    print("\n           ===== Results =====")
    print(f"           Accuracy : {results['accuracy']:.8f}")
    print(f"           F1-score : {results['f1']:.8f}")
    print(f"           Precision: {results['precision']:.8f}")
    print(f"           Recall   : {results['recall']:.8f}")