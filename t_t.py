import pandas as pd

# =====================================================
# CARGA DATASET PROCESADO
# =====================================================
df = pd.read_csv("dataset_procesado.csv")

# =====================================================
# SPLIT TEMPORAL (SIN SHUFFLE)
# NO ROMPE LOS DELTAS
# =====================================================
ratio_train = 0.8
split_idx = int(len(df) * ratio_train)

train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df  = df.iloc[split_idx:].reset_index(drop=True)

# =====================================================
# GUARDADO
# =====================================================
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("✔ Split temporal aplicado")
print("✔ Train filas:", len(train_df))
print("✔ Test filas :", len(test_df))