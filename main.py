import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
from Parser import Parser
from aeon.datasets import load_regression
from aeon.transformations.collection.feature_based import Catch22
from aeon.transformations.collection.convolution_based import MiniRocket
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
from aeon.regression.interval_based import DrCIFRegressor
from utils import get_data, resample_data, load_experiment_data, adjust_ts_length, to_tsfresh_df

def main():
    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract config parameters.
    NUM_RUNS = int(config.get("runs", 1))
    DATASETS = config.get("datasets", [])
    MODELS = config.get("models", [])
    FILENAME = config.get("filename", 'teste')
    SEED = config.get("seed", 0)
    DEVICE =  config.get("device", 'cpu')
    DATA_PATH = config.get("data_path", 'data/')

    csv_path = os.path.join("./outputs", FILENAME)

    for run in range(NUM_RUNS):
        print(f'\n === Started run {run+1} === ')

        for dataset_name in DATASETS:
            print(f'\n    --- Executing dataset {dataset_name} --- ')

            for model in MODELS:
                print(f'\n        -> Running model {model}  ')

                X_train, y_train, X_test, y_test = get_data(DATA_PATH, dataset_name, SEED+run)

                if model == 'c22':
                    catch22 = Catch22()
                    X_train = catch22.fit_transform(X_train)
                    X_test = catch22.transform(X_test)

                elif model == 'tsfresh':
                    base_path = f'./tsfresh_data/{dataset_name}'
                    os.makedirs(base_path, exist_ok=True)

                    train_path = os.path.join(base_path, 'X_train.csv')
                    test_path  = os.path.join(base_path, 'X_test.csv')

                    if os.path.exists(train_path) and os.path.exists(test_path):
                        X_train = pd.read_csv(train_path, index_col=0).to_numpy()
                        X_test  = pd.read_csv(test_path, index_col=0).to_numpy()
                    else:
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
                        print('Extraindo features treino...')
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

                        X_train.to_csv(train_path)
                        X_test.to_csv(test_path)

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

                elif model == 'minirocket_fm':
                    minirocket = MiniRocket()
                    minirocket.fit(X_train)

                    X_train_feat = minirocket.transform(X_train)
                    X_test_feat = minirocket.transform(X_test)

                    X_train = np.asarray(X_train_feat)
                    X_test = np.asarray(X_test_feat)

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
    
    print('\nDONE!')

if __name__ == "__main__":
    main()