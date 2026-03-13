import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
from utils.Parser import Parser
from utils.utils_task import choose_functions
from aeon.transformations.collection.feature_based import Catch22
import time
import csv

def main():
    parser = Parser()
    config = parser.parse()

    DATASETS = config.get("datasets", [])
    TASK = config.get("task", None)
    DATA_PATH = TASK + '/' + config.get("data_path", 'data/')
    FEATURE_DIR = TASK + "/feature_cache_catch22"
    os.makedirs(FEATURE_DIR, exist_ok=True)

    TIME_LOG_FILE = './tsc/outputs/time_catch22.csv'

    for dataset_name in DATASETS:

        print(f"\n=== Extracting Catch22 features for {dataset_name} ===")

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

            print("Initializing Catch22...")
            transformer = Catch22(n_jobs=16)

            print("Fitting Catch22...")
            start_time = time.time()
            transformer.fit(X_train)

            print("Transforming dataset...")
            X_full_feat = transformer.transform(X_full)
            end_time = time.time()

            duration = end_time - start_time

            if not isinstance(X_full_feat, np.ndarray):
                X_full_feat = X_full_feat.to_numpy()

            # print("Saving features...")

            # np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_X.npy"), X_full_feat)
            # np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_y.npy"), y_full)
            # np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_ntrain.npy"), np.array([n_train]))

            file_exists = os.path.isfile(TIME_LOG_FILE)
            with open(TIME_LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)

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