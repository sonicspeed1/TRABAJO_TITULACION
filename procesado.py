import pandas as pd
import numpy as np

# =====================================================
# 1. CARGA DE DATOS ORIGINALES
# =====================================================
df = pd.read_csv("datos_total.csv")

# =====================================================
# 2. CONVERSIÓN A DATETIME (SOLO REFERENCIA TEMPORAL)
# =====================================================
df["FechaHora"] = pd.to_datetime(
    df["Fecha y Hora de Registro"],
    errors="coerce"
)

# Eliminar registros con fecha inválida
df = df.dropna(subset=["FechaHora"])

# Orden temporal estricto (CRÍTICO)
df = df.sort_values("FechaHora").reset_index(drop=True)

# =====================================================
# 3. IDENTIFICACIÓN DE FECHA CALENDARIO
# =====================================================
df["Dia"] = df["FechaHora"].dt.date

# Detectar cambio real de día (00:00)
cambio_dia = df["Dia"] != df["Dia"].shift(1)

# =====================================================
# 4. SUAVIZADO DE VELOCIDAD (ANTI-RUIDO GPS)
# =====================================================
# Media móvil por día para no mezclar jornadas
df["Velocidad_f"] = (
    df.groupby("Dia")["Velocidad (Km/h)"]
      .transform(lambda x: x.rolling(
          window=3,            # Ventana pequeña (no destruye dinámica)
          center=True,
          min_periods=1
      ).mean())
)

# =====================================================
# 5. DELTA DE TIEMPO (SEGUNDOS)
# =====================================================
# Diferencia real entre observaciones
df["delta_t_s"] = df["FechaHora"].diff().dt.total_seconds()

# Reinicio SOLO cuando cambia el día
df.loc[cambio_dia, "delta_t_s"] = 0

# Rellenar primer valor
df["delta_t_s"] = df["delta_t_s"].fillna(0)

# Limitar valores extremos (ej: pausas largas)
df["delta_t_s"] = df["delta_t_s"].clip(0, 300)

# Escala logarítmica para estabilidad numérica
df["delta_t_s"] = np.log1p(df["delta_t_s"])

# =====================================================
# 6. DELTA DE VELOCIDAD (DINÁMICA REAL)
# =====================================================
df["delta_v"] = df["Velocidad_f"].diff()

# Reinicio diario
df.loc[cambio_dia, "delta_v"] = 0
df["delta_v"] = df["delta_v"].fillna(0)

# Limitar cambios físicamente imposibles
df["delta_v"] = df["delta_v"].clip(-30, 30)

# =====================================================
# 7. DELTAS ESPACIALES (MOVIMIENTO REAL)
# =====================================================
df["delta_lat"] = df["Latitud"].diff()
df["delta_lon"] = df["Longitud"].diff()

# Reinicio diario
df.loc[cambio_dia, ["delta_lat", "delta_lon"]] = 0

# Rellenar nulos
df[["delta_lat", "delta_lon"]] = df[["delta_lat", "delta_lon"]].fillna(0)

# Limitar saltos GPS irreales (~100m)
df["delta_lat"] = df["delta_lat"].clip(-0.001, 0.001)
df["delta_lon"] = df["delta_lon"].clip(-0.001, 0.001)

# =====================================================
# 8. LIMPIEZA FINAL
# =====================================================
df = df.drop(columns=[
    "Velocidad_f"   # Intermedia, no se usa como feature directa
])

# =====================================================
# 9. GUARDAR DATASET PROCESADO
# =====================================================
df.to_csv("dataset_procesado.csv", index=False)

print("✔ Preprocesamiento finalizado correctamente")
print("✔ Deltas continuos dentro del día")
print("✔ Reinicio SOLO por cambio de fecha")
print("✔ Señal temporal estabilizada para LSTM")