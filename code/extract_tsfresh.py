import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from utils.Parser import Parser
from utils.utils_task import choose_functions
import time
import csv

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

def main():
    parser = Parser()
    config = parser.parse()

    DATASETS = config.get("datasets", [])
    TASK = config.get("task", None)
    DATA_PATH = TASK + '/' + config.get("data_path", 'data/')
    FEATURE_DIR = TASK + "/feature_cache"
    os.makedirs(FEATURE_DIR, exist_ok=True)
    TIME_LOG_FILE = './tsc/outputs/time_tsfresh.csv'

    for dataset_name in DATASETS:

        print(f"\n=== Extracting features for {dataset_name} ===")

        # IMPORTANT: resample_id=0 disables shuffling
        get_data, _, _, _, _ = choose_functions(TASK)
        X_train, y_train, X_test, y_test = get_data(
            DATA_PATH,
            dataset_name,
            resample_id=0
        )

        n_train = len(X_train)

        try:
            # Merge in original order
            X_full = np.concatenate([X_train, X_test], axis=0)
            y_full = np.concatenate([y_train, y_test], axis=0)

            print("Converting to tsfresh format...")
            df_full = to_tsfresh_df(X_full)

            print("Extracting tsfresh features...")
            start_time = time.time()
            X_full_feat = extract_features(
                df_full,
                column_id="id",
                column_sort="time",
                column_kind="kind",
                column_value="value",
                default_fc_parameters=EfficientFCParameters(),
                # n_jobs=max(1, os.cpu_count() // 2)
                n_jobs=16
            )
            end_time = time.time()
            duration = end_time - start_time

            impute(X_full_feat)
            X_full_feat = X_full_feat.to_numpy()

            print("Saving features...")
            np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_X.npy"), X_full_feat)
            np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_y.npy"), y_full)
            np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_ntrain.npy"), np.array([n_train]))

            file_exists = os.path.isfile(TIME_LOG_FILE)
            with open(TIME_LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                # Se o arquivo for novo, escreve o cabeçalho
                if not file_exists:
                    writer.writerow(["dataset", "n_samples", "n_features", "time_seconds"])
                
                writer.writerow([
                    dataset_name, 
                    len(X_full), 
                    X_full_feat.shape[1], 
                    round(duration, 4)
                ])

            print(f"Done for {dataset_name}")
        
        except Exception as e:
                    print(f"[ERROR] Dataset {dataset_name} failed:")
                    print(type(e).__name__, e)

if __name__ == "__main__":
    main()
