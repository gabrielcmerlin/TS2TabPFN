import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
import pandas as pd
from Parser import Parser
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from utils_data import get_data
from utils_train import to_tsfresh_df

def main():
    parser = Parser()
    config = parser.parse()

    DATASETS = config.get("datasets", [])
    DATA_PATH = config.get("data_path", "data/")

    FEATURE_DIR = "./feature_cache"
    os.makedirs(FEATURE_DIR, exist_ok=True)

    for dataset_name in DATASETS:

        print(f"\n=== Extracting features for {dataset_name} ===")

        # IMPORTANT: resample_id=0 disables shuffling
        X_train, y_train, X_test, y_test = get_data(
            DATA_PATH,
            dataset_name,
            resample_id=0
        )

        n_train = len(X_train)

        # Merge in original order
        X_full = np.concatenate([X_train, X_test], axis=0)
        y_full = np.concatenate([y_train, y_test], axis=0)

        print("Converting to tsfresh format...")
        df_full = to_tsfresh_df(X_full)

        print("Extracting tsfresh features...")
        X_full_feat = extract_features(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=max(1, os.cpu_count() // 2)
        )

        impute(X_full_feat)
        X_full_feat = X_full_feat.to_numpy()

        print("Saving features...")

        np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_X.npy"), X_full_feat)
        np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_y.npy"), y_full)
        np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_ntrain.npy"), np.array([n_train]))

        print(f"Done for {dataset_name}")


if __name__ == "__main__":
    main()