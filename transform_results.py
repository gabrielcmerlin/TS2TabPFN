import pandas as pd
import re

# ========= CONFIG =========
input_csv = "outputs/c22.csv"
output_csv = "outputs/c22_folds.csv"
metric = "rmse"
# ==========================

# Ler CSV original
df = pd.read_csv(input_csv)

# Extrair número do run (run1 -> 0, run2 -> 1, ...)
df["fold"] = df["run"].apply(lambda x: int(re.search(r"run(\d+)", x).group(1)) - 1)

# Pivotar para formato wide
pivot = df.pivot(index="dataset", columns="fold", values=metric)

# Ordenar colunas (folds)
pivot = pivot.sort_index(axis=1)

# Garantir 30 folds (0–29)
for i in range(30):
    if i not in pivot.columns:
        pivot[i] = None

pivot = pivot[sorted(pivot.columns)]

# Resetar index para dataset virar coluna normal
pivot = pivot.reset_index()

# Renomear colunas para algo mais explícito (opcional)
pivot.columns = ["dataset"] + [f"{i}" for i in range(30)]

# Salvar como CSV
pivot.to_csv(output_csv, index=False)

print("Arquivo convertido salvo em:", output_csv)