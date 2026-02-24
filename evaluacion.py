import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================
# CARGA DE RESULTADOS
# =====================================================
df = pd.read_csv("predicciones.csv")

y_real = df["Velocidad (Km/h)"].values
y_pred = df["Vel_Pred_kmh"].values

# =====================================================
# MÉTRICAS
# =====================================================
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
mae = mean_absolute_error(y_real, y_pred)
r2 = r2_score(y_real, y_pred)

print("Evaluación del modelo")
print(f"RMSE: {rmse:.3f} km/h")
print(f"MAE : {mae:.3f} km/h")
print(f"R²  : {r2:.3f}")