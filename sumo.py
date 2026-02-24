import pandas as pd
import traci
import sumolib
import math

# ===============================
# 1. CARGAR PREDICCIONES
# ===============================
df = pd.read_csv("predicciones.csv")

# ===============================
# 2. INICIAR SUMO
# ===============================
net = sumolib.net.readNet("quito-sur.net.xml")

sumoCmd = [
    "sumo-gui",
    "-c", "simulation.sumo.cfg",
    "--step-length", "1",
    "--start",
    "--emission-output", "emisiones.xml"
]

traci.start(sumoCmd)

# ===============================
# 3. RUTA BASE (DUMMY)
# ===============================
edges = [list(net.getEdges())[0].getID()]
traci.route.add("r0", edges)

# ===============================
# 4. CREAR TODOS LOS VEHÍCULOS
# ===============================
veh_ids = sorted(df["Nº"].unique())

for vid in veh_ids:
    traci.vehicle.add(
        vehID=f"veh_{int(vid)}",
        routeID="r0",
        typeID="DEFAULT_VEHTYPE",
        depart=0
    )

# Avanzar hasta que TODOS existan
while len(traci.vehicle.getIDList()) < len(veh_ids):
    traci.simulationStep()

# ===============================
# 5. SIMULACIÓN PRINCIPAL
# ===============================
sumo_results = []
step_global = 0

for _, row in df.iterrows():

    vehID = f"veh_{int(row['Nº'])}"

    lat = row["Latitud"]
    lon = row["Longitud"]

    speed_real = row["Velocidad (Km/h)"]
    speed_pred = row["Vel_Pred_kmh"]
    speed_used = speed_pred if not pd.isna(speed_pred) else speed_real

    delta_t = max(1, int(round(row["delta_t_s"])))

    x, y = net.convertLonLat2XY(lon, lat)

    # ===============================
    # ESPERAR VEHÍCULO (CLAVE)
    # ===============================
    while vehID not in traci.vehicle.getIDList():
        traci.simulationStep()
        step_global += 1

    # ===============================
    # ACTUALIZAR VEHÍCULO
    # ===============================
    traci.vehicle.moveToXY(
        vehID,
        edgeID="",
        laneIndex=0,
        x=x,
        y=y,
        keepRoute=2
    )

    traci.vehicle.setSpeed(
        vehID,
        max(speed_used, 0) / 3.6
    )

    # ===============================
    # AVANZAR TIEMPO REAL
    # ===============================
    for _ in range(delta_t):
        traci.simulationStep()
        step_global += 1

        speed_sumo = traci.vehicle.getSpeed(vehID) * 3.6
        pos_x, pos_y = traci.vehicle.getPosition(vehID)
        co2 = traci.vehicle.getCO2Emission(vehID)

        sumo_results.append([
            step_global,
            vehID,
            lat,
            lon,
            speed_sumo,
            speed_used,
            speed_real,
            speed_used - speed_real,
            co2,
            pos_x,
            pos_y,
            len(veh_ids)
        ])

# ===============================
# 6. CERRAR SUMO
# ===============================
traci.close()

# ===============================
# 7. GUARDAR RESULTADOS
# ===============================
sumo_df = pd.DataFrame(
    sumo_results,
    columns=[
        "Step",
        "Veh_ID",
        "Latitud",
        "Longitud",
        "Vel_SUMO_kmh",
        "Vel_Usada_kmh",
        "Vel_Real_kmh",
        "Error_kmh",
        "CO2_mg_s",
        "X",
        "Y",
        "Flujo_vehicular"
    ]
)

sumo_df.to_csv("resultados_sumo_estable.csv", index=False)

print("Simulación finalizada SIN errores")
print("Resultados densos listos para graficar")
