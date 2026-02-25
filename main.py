import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import time
from Parser import Parser
from utils_data import get_data_tsfresh
from utils_train import store_results, extract_feat
from utils_task import choose_functions

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
    TASK = config.get("task", None)

    csv_path = os.path.join("./outputs", FILENAME)

    get_data, train_model, get_results, print_results = choose_functions(TASK)

    for run in range(NUM_RUNS):
        print(f'\n === Started run {run+1} === ')

        for dataset_name in DATASETS:
            print(f'\n    --- Executing dataset {dataset_name} --- ')

            for model_name in MODELS:
                print(f'\n        -> Running model {model_name}  ')

                if model_name == "tsfresh":
                    X_train, y_train, X_test, y_test = get_data_tsfresh("./feature_cache_reg",dataset_name,SEED+run)
                else:
                    X_train, y_train, X_test, y_test = get_data(DATA_PATH, dataset_name, SEED+run)
                    X_train, X_test = extract_feat(X_train, X_test, model_name)

                try:
                    start_time = time.time()
                    y_pred = train_model(X_train, X_test, y_train, SEED, run, DEVICE, model_name)
                    elapsed_time = time.time() - start_time

                    results = get_results(y_test, y_pred, model_name, dataset_name, elapsed_time, run)
                    print_results(results)
                    store_results(results, csv_path)

                except Exception as e:
                    print(f"[ERROR] Model {model_name} on dataset {dataset_name} failed:")
                    print(type(e).__name__, e)
    
    print('\nDONE!')

if __name__ == "__main__":
    main()