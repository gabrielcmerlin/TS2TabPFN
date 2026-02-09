import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from Parser import Parser
from aeon.datasets import load_regression
from aeon.transformations.collection.feature_based import Catch22
from tabpfn import TabPFNRegressor
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import os
import time
from datetime import datetime

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

def get_data(dataset_name, model_name):
    # Download data.
    X_train, y_train = load_regression(dataset_name, split="train")
    X_test, y_test = load_regression(dataset_name, split="test")

    # Get rid of NaNs.
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Normalize per time series.
    X_train = (X_train - X_train.mean(axis=2, keepdims=True)) / (
        X_train.std(axis=2, keepdims=True) + 1e-8
    )
    X_test = (X_test - X_test.mean(axis=2, keepdims=True)) / (
        X_test.std(axis=2, keepdims=True) + 1e-8
    )

    if model_name == 'raw':
        # Caso: (N, C, T)
        if X_train.ndim == 3:
            N, C, T = X_train.shape

            if C == 1:
                # Remove canal singleton -> (N, T)
                X_train = X_train[:, 0, :]
                X_test  = X_test[:, 0, :]
            else:
                # Concatena canais -> (N, C*T)
                X_train = X_train.reshape(N, C * T)
                X_test  = X_test.reshape(X_test.shape[0], C * T)

    return X_train, y_train, X_test, y_test

def main():
    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract config parameters.
    NUM_RUNS = int(config.get("runs", 1))
    DATASETS = config.get("datasets", [])
    MODELS = config.get("models", [])
    FILENAME = config.get("filename", 'teste')
    SEED = config.get("seed", 42)
    DEVICE =  config.get("device", 'cpu')

    csv_path = os.path.join("./outputs", FILENAME)

    for run in range(NUM_RUNS):
        print(f'\n === Started run {run+1} === ')

        for dataset_name in DATASETS:
            print(f'\n    --- Executing dataset {dataset_name} --- ')

            for model in MODELS:
                print(f'\n        -> Running model {model}  ')

                X_train, y_train, X_test, y_test = get_data(dataset_name, model)

                if model == 'c22':
                    catch22 = Catch22()
                    X_train = catch22.fit_transform(X_train)
                    X_test = catch22.transform(X_test)

                elif model == 'tsfresh':
                    df_train = to_tsfresh_df(X_train)
                    df_test  = to_tsfresh_df(X_test)

                    fc_params = EfficientFCParameters()
                    X_train = extract_features(
                        df_train,
                        column_id="id",
                        column_sort="time",
                        column_kind="kind",
                        column_value="value",
                        default_fc_parameters=fc_params,
                        disable_progressbar=True
                    )
                    X_test = extract_features(
                        df_test,
                        column_id="id",
                        column_sort="time",
                        column_kind="kind",
                        column_value="value",
                        default_fc_parameters=fc_params,
                        disable_progressbar=True
                    )

                    impute(X_train)
                    impute(X_test)
                    X_train = X_train.to_numpy()
                    X_test  = X_test.to_numpy()

                elif model == 'mantis':
                    network = Mantis8M(device='cuda')
                    network = network.from_pretrained("paris-noah/Mantis-8M")
                    mantis = MantisTrainer(device='cuda', network=network)
                    num_patches = network.tokgen_unit.num_patches

                    X_train = adjust_ts_length(X_train, num_patches)
                    X_test  = adjust_ts_length(X_test, num_patches)

                    X_train = mantis.transform(X_train)
                    X_test = mantis.transform(X_test)

                try:
                    start_time = time.time()
                    regressor = TabPFNRegressor(
                        random_state=SEED+run,
                        ignore_pretraining_limits=True,
                        device=DEVICE
                    )
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    elapsed_time = time.time() - start_time

                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    print("\n           ===== Results =====")
                    print(f"           MSE : {mse:.8f}")
                    print(f"           MAE : {mae:.8f}")
                    print(f"           RMSE: {rmse:.8f}")
                    print(f"           R²  : {r2:.8f}")

                    row = {
                        "model": model,
                        "dataset": dataset_name,
                        "mse": mse,
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                        "time": elapsed_time,
                        "run": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run+1}"
                    }

                    df_row = pd.DataFrame([row])

                    # Append no CSV
                    if os.path.exists(csv_path):
                        df_row.to_csv(csv_path, mode="a", header=False, index=False)
                    else:
                        df_row.to_csv(csv_path, index=False)

                except Exception as e:
                    print(f"[ERROR] Model {model} on dataset {dataset_name} failed:")
                    print(type(e).__name__, e)

if __name__ == "__main__":
    main()