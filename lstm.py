import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# CARGA TRAIN
# =====================================================
df = pd.read_csv("train.csv")

features = ["delta_v", "delta_lat", "delta_lon", "delta_t_s"]
target = "delta_v"

X = df[features].values
y = df[target].values.reshape(-1, 1)

# =====================================================
# ESCALADO ÚNICO (MUY IMPORTANTE)
# =====================================================
scaler = MinMaxScaler()
XY_scaled = scaler.fit_transform(np.hstack([X, y]))

X_scaled = XY_scaled[:, :-1]
y_scaled = XY_scaled[:, -1].reshape(-1, 1)

joblib.dump(scaler, "scaler.pkl")

# =====================================================
# SECUENCIAS
# =====================================================
VENTANA = 20

Xs, ys = [], []
for i in range(len(X_scaled) - VENTANA):
    Xs.append(X_scaled[i:i+VENTANA])
    ys.append(y_scaled[i+VENTANA])

X_seq = np.array(Xs)
y_seq = np.array(ys)

# =====================================================
# MODELO
# =====================================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(VENTANA, X_seq.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early = EarlyStopping(patience=8, restore_best_weights=True)

model.fit(
    X_seq,
    y_seq,
    epochs=100,
    batch_size=32,
    callbacks=[early],
    verbose=1
)

model.save("modelo_lstm.keras")

print("✔ Modelo entrenado correctamente")