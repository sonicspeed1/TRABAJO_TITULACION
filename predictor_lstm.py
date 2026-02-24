import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

df = pd.read_csv("test.csv")

features = ["delta_v", "delta_lat", "delta_lon", "delta_t_s"]

model = tf.keras.models.load_model("modelo_lstm.keras")
scaler = joblib.load("scaler.pkl")

X = df[features].values
dummy_y = np.zeros((len(X), 1))

XY = np.hstack([X, dummy_y])
XY_scaled = scaler.transform(XY)
X_scaled = XY_scaled[:, :-1]

# Secuencias
VENTANA = 20
X_seq = np.array([
    X_scaled[i:i+VENTANA]
    for i in range(len(X_scaled) - VENTANA)
])

# Predicción delta_v
delta_v_scaled = model.predict(X_seq)
dummy = np.zeros((len(delta_v_scaled), X.shape[1] + 1))
dummy[:, -1] = delta_v_scaled[:, 0]

delta_v_pred = scaler.inverse_transform(dummy)[:, -1]

# Reconstrucción velocidad
df_pred = df.iloc[VENTANA:].copy()
df_pred["Vel_Pred_kmh"] = (
    df_pred["Velocidad (Km/h)"].values + delta_v_pred
)

df_pred.to_csv("predicciones.csv", index=False)

print("✔ Predicción cinemática generada")