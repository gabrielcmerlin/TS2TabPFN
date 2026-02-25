import pandas as pd
import re

# ========= CONFIG =========
input_csv = "outputs/tsfresh_eff.csv"
output_csv = "results/Tab_tsfresh.csv"
metric = "rmse"
# ==========================

# ==========================
# 1) Ler CSV
# ==========================
df = pd.read_csv(input_csv)

# ==========================
# 2) Ordenar por timestamp (garante ordem real de execução)
# ==========================
df = df.sort_values("run")

# ==========================
# 3) Reindexar folds manualmente por dataset
# ==========================

# Lista para guardar contagem de cada dataset
dataset_counts = {}

# Lista que vai virar a nova coluna fold
new_folds = []

for _, row in df.iterrows():
    dataset = row["dataset"]
    
    # Se ainda não vimos esse dataset
    if dataset not in dataset_counts:
        dataset_counts[dataset] = 0
    
    # Atribui fold atual
    new_folds.append(dataset_counts[dataset])
    
    # Incrementa contador
    dataset_counts[dataset] += 1

# Substitui coluna fold
df["fold"] = new_folds

# ==========================
# 4) Pivot seguro
# ==========================
pivot = df.pivot(
    index="dataset",
    columns="fold",
    values=metric
)

# ==========================
# 5) Garantir 30 folds (0–29)
# ==========================
for i in range(30):
    if i not in pivot.columns:
        pivot[i] = None

pivot = pivot[sorted(pivot.columns)]

# ==========================
# 6) Resetar index
# ==========================
pivot = pivot.reset_index()

pivot.columns = ["dataset"] + [f"{i}" for i in range(30)]

# ==========================
# 7) Salvar
# ==========================
pivot.to_csv(output_csv, index=False)

print("Arquivo convertido salvo em:", output_csv)