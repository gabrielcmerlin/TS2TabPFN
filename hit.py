import os
import torch
import numpy as np
import pandas as pd
import random
from Parser import Parser
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from aeon.datasets import load_regression
import time
from datetime import datetime

from model import HinceptionTime

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def get_data(dataset_name):
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

    return X_train, y_train, X_test, y_test

def main():

    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract config parameters.
    NUM_RUNS = int(config.get("runs", 1))
    DATASETS = config.get("datasets", [])
    FILENAME = config.get("filename", 'teste')
    SEED = config.get("seed", 42)
    DEVICE =  config.get("device", 'cpu')

    device = DEVICE
    batch_size = 32
    epochs = 2000
    num_models = 5

    csv_path = os.path.join("./outputs", FILENAME)

    for run in range(NUM_RUNS):
        set_seed(SEED+run)
        print(f'\n === Started run {run+1} === ')

        for dataset_name in DATASETS:
            print(f'\n    --- Executing dataset {dataset_name} --- ')

            X_train, y_train, X_test, y_test = get_data(dataset_name)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            sequence_length = X_train.shape[-1]
            in_channels = X_train.shape[1]

            configs = {
                "loss": MSELoss(),
                "optimizer": Adam,
                "lr": 0.001,
            }

            start_time = time.time()

            model = HinceptionTime(
                configs=configs,
                sequence_length=sequence_length,
                in_channels=in_channels,
                num_models=num_models,
                seed=SEED+run
            )

            model.run(
                train_loader=train_loader,
                epochs=epochs,
                device=device,
            )

            model.to(device)
            model.eval()

            y_pred = []
            y_test = []

            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    preds = model.forward(x)
                    y_pred.extend(preds.cpu().numpy())
                    y_test.extend(y.numpy())

            y_pred = np.array(y_pred).flatten()
            y_test = np.array(y_test).flatten()

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
                "model": 'HIT',
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

    print('DONE!')

if __name__ == "__main__":
    main()