import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
from Parser import Parser
from utils_task import choose_functions
from aeon.transformations.collection.convolution_based import MultiRocket

def main():
    parser = Parser()
    config = parser.parse()

    DATASETS = config.get("datasets", [])
    TASK = config.get("task", None)
    DATA_PATH = TASK + '/' + config.get("data_path", 'data/')
    FEATURE_DIR = TASK + "/feature_cache_multi"
    os.makedirs(FEATURE_DIR, exist_ok=True)

    for dataset_name in DATASETS:

        print(f"\n=== Extracting MultiROCKET features for {dataset_name} ===")

        # IMPORTANT: resample_id=0 disables shuffling
        get_data, _, _, _ = choose_functions(TASK)
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

            print("Initializing MultiROCKET...")
            transformer = MultiRocket()

            print("Fitting MultiROCKET on training data...")
            transformer.fit(X_train)

            print("Transforming full dataset...")
            X_full_feat = transformer.transform(X_full)

            # Convert to numpy if needed
            if not isinstance(X_full_feat, np.ndarray):
                X_full_feat = X_full_feat.to_numpy()

            print("Saving features...")

            np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_X.npy"), X_full_feat)
            np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_y.npy"), y_full)
            np.save(os.path.join(FEATURE_DIR, f"{dataset_name}_ntrain.npy"), np.array([n_train]))

            print(f"Done for {dataset_name}")

        except Exception as e:
            print(f"[ERROR] Dataset {dataset_name} failed:")
            print(type(e).__name__, e)


if __name__ == "__main__":
    main()