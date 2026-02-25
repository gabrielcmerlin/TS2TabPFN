import os
import numpy as np
import pandas as pd
from datetime import datetime
from aeon.transformations.collection.feature_based import Catch22
from tsfresh import extract_features
from tabpfn import TabPFNRegressor
from aeon.regression.convolution_based import MiniRocketRegressor
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
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
    
    if model == 'c22':
        catch22 = Catch22()
        X_train = catch22.fit_transform(X_train)
        X_test = catch22.transform(X_test)

    elif model == 'tsfresh':
        df_train = to_tsfresh_df(X_train)
        df_test  = to_tsfresh_df(X_test)
        fc_params = EfficientFCParameters()
        n_jobs = max(1, os.cpu_count() // 2)

        print('Extraindo features treino...')
        X_train = extract_features(
            df_train,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=fc_params,
            n_jobs=n_jobs
        )
        print('Extraindo features teste...')
        X_test = extract_features(
            df_test,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=fc_params,
            n_jobs=n_jobs
        )

        impute(X_train)
        impute(X_test)
        X_train = X_train.to_numpy()
        X_test  = X_test.to_numpy()

    elif model == 'multirocket_fm':
        multirocket = MultiRocket()
        multirocket.fit(X_train)

        X_train_feat = multirocket.transform(X_train)
        X_test_feat = multirocket.transform(X_test)

        X_train = np.asarray(X_train_feat)
        X_test = np.asarray(X_test_feat)

    return X_train, X_test

def train_model(X_train, X_test, y_train, SEED, run, DEVICE, model_name):
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
    
def get_results(y_test, y_pred, model_name, dataset_name, elapsed_time, run):
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

def store_results(results, csv_path):
    df_row = pd.DataFrame([results])

    # Append no CSV
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)