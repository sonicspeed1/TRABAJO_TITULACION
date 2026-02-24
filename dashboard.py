import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import time

# ==================================================
# CONFIGURACIÓN GENERAL
# ==================================================
st.set_page_config(page_title="Panel de Análisis Vehicular", layout="wide")
st.title("Panel Inteligente de Análisis Vehicular")

st.sidebar.title("Tipo de análisis")
tipo_analisis = st.sidebar.radio(
    "Selecciona qué deseas evaluar:",
    ["Velocidad (Modelo LSTM)", "Emisiones CO₂"]
)

# ==================================================
# MODO VELOCIDAD
# ==================================================
if tipo_analisis == "Velocidad (Modelo LSTM)":

    st.header("Comparación entre velocidad real y velocidad predicha")

    @st.cache_data
    def load_speed():
        df = pd.read_csv("predicciones.csv")
        df["FechaHora"] = pd.to_datetime(df["FechaHora"], errors="coerce")
        df = df.dropna(subset=["Vel_Pred_kmh", "Velocidad (Km/h)", "FechaHora"])
        return df.sort_values("FechaHora")

    df = load_speed()

    st.sidebar.subheader("Configuración de velocidad")
    modo = st.sidebar.selectbox("Modo de análisis", ["Por horas (1 día)", "Comparación por días"])
    fecha_inicio = st.sidebar.date_input("Fecha inicio", value=df["FechaHora"].dt.date.min())
    fecha_fin = st.sidebar.date_input("Fecha fin", value=df["FechaHora"].dt.date.max())
    hora_inicio = st.sidebar.time_input("Hora inicio", value=time(0, 0))
    hora_fin = st.sidebar.time_input("Hora fin", value=time(23, 59))
    generar = st.sidebar.button("Generar análisis")

    if generar:
        df_f = df[(df["FechaHora"].dt.date >= fecha_inicio) &
                  (df["FechaHora"].dt.date <= fecha_fin)]

        if modo == "Por horas (1 día)":
            df_f = df_f[(df_f["FechaHora"].dt.time >= hora_inicio) &
                        (df_f["FechaHora"].dt.time <= hora_fin)]

        if df_f.empty:
            st.warning("No hay datos para el rango seleccionado.")
            st.stop()

        if modo == "Por horas (1 día)":
            df_grp = df_f.set_index("FechaHora").resample("H").mean(numeric_only=True).reset_index()
            tick_format = "%H:%M\n%d-%m-%Y"
            titulo_x = "Hora"
        else:
            df_grp = df_f.set_index("FechaHora").resample("D").mean(numeric_only=True).reset_index()
            tick_format = "%d-%m-%Y"
            titulo_x = "Día"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_grp["FechaHora"],
            y=df_grp["Velocidad (Km/h)"],
            mode="lines",
            line=dict(color="rgb(31,119,180)", width=4),
            name="Velocidad real"
        ))

        fig.add_trace(go.Scatter(
            x=df_grp["FechaHora"],
            y=df_grp["Vel_Pred_kmh"],
            mode="lines",
            line=dict(color="rgb(214,39,40)", width=4),
            name="Velocidad predicha"
        ))

        fig.update_layout(
            title=dict(text="Comparación entre velocidad real y velocidad predicha",
                       x=0.5, font=dict(size=22, family="Arial", color="black")),
            xaxis=dict(title=dict(text=titulo_x, font=dict(size=18, family="Arial Black")),
                       tickformat=tick_format, tickangle=-25,
                       showgrid=True, gridcolor="#d9d9d9",
                       showline=True, linecolor="black", linewidth=2, mirror=True),
            yaxis=dict(title=dict(text="Velocidad (km/h)", font=dict(size=18, family="Arial Black")),
                       showgrid=True, gridcolor="#d9d9d9",
                       showline=True, linecolor="black", linewidth=2, mirror=True),
            legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1,
                        font=dict(size=14, family="Arial Black")),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=90, r=40, t=90, b=90),
            height=550
        )

        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# MODO CO2
# ==================================================
else:

    st.header(" Emisiones promedio de CO₂ por vehículo")

    @st.cache_data
    def load_co2():
        df = pd.read_csv("resultados_sumo_final.csv")
        df = df.dropna(subset=["Veh_ID", "CO2_mg_s"])
        return df

    df = load_co2()

    st.sidebar.subheader("Selección de vehículos")
    vehiculos = sorted(df["Veh_ID"].unique())
    vehiculos_sel = st.sidebar.multiselect("Selecciona vehículos", vehiculos, default=vehiculos[:3])
    generar = st.sidebar.button("Generar comparación")

    if generar:
        df_f = df[df["Veh_ID"].isin(vehiculos_sel)]
        df_avg = df_f.groupby("Veh_ID", as_index=False).agg(CO2_promedio=("CO2_mg_s", "mean"))

        col1, col2 = st.columns(2)
        col1.metric("CO₂ promedio global", f"{df_avg['CO2_promedio'].mean():.2f}")
        col2.metric("Vehículos comparados", len(vehiculos_sel))

        colores = ["rgb(44,162,95)" if i % 2 == 0 else "rgb(31,119,180)" for i in range(len(df_avg))]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_avg["Veh_ID"],
            y=df_avg["CO2_promedio"],
            text=[f"{v:.2f}" for v in df_avg["CO2_promedio"]],
            textposition="inside",
            textfont=dict(size=18, color="black", family="Arial Black"),
            marker=dict(color=colores)
        ))

        fig.update_layout(
            title=dict(text="Emisiones promedio de CO₂ por vehículo",
                       x=0.5, font=dict(size=22, family="Arial")),
            xaxis=dict(title=dict(text="Vehículo", font=dict(size=18, family="Arial Black")),
                       showline=True, linecolor="black", linewidth=2, mirror=True),
            yaxis=dict(title=dict(text="CO₂ promedio (mg/s)", font=dict(size=18, family="Arial Black")),
                       showgrid=True, gridcolor="#d9d9d9",
                       showline=True, linecolor="black", linewidth=2, mirror=True),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=90, r=40, t=90, b=90),
            height=550
        )

        st.plotly_chart(fig, use_container_width=True)
