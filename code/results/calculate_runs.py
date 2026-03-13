import pandas as pd

path = "tsc/outputs/multi.csv"

df = pd.read_csv(path)  # agora o header é interpretado corretamente

runs_por_dataset = df.groupby("dataset").size()

diff_30 = runs_por_dataset[runs_por_dataset != 30].sort_index()

print("Datasets com número de runs diferente de 30:\n")
print(diff_30.to_string())

print("\nResumo:")
print(f"Datasets com 30 runs: {(runs_por_dataset == 30).sum()}")
print(f"Datasets com número diferente de 30 runs: {(runs_por_dataset != 30).sum()}")
print(f"Total de datasets: {runs_por_dataset.shape[0]}")