import pandas as pd
import traci
import sumolib

# ===============================
# 1. CARGAR DATOS
# ===============================
df = pd.read_csv("predicciones.csv")
df["Veh_ID"] = df["Nº"].astype(int)

# ===============================
# 2. INICIAR SUMO (SIN GUI)
# ===============================
net = sumolib.net.readNet("quito-sur.net.xml")

traci.start([
    "sumo",
    "-c", "simulation.sumo.cfg",
    "--emission-output", "emisiones.xml"
])

# Paso inicial obligatorio
traci.simulationStep()

# ===============================
# 3. RUTA BASE
# ===============================
edges = list(net.getEdges())
traci.route.add("r0", [edges[0].getID()])

# ===============================
# 4. CONTENEDORES
# ===============================
sumo_results = []

# ===============================
# 5. LOOP PRINCIPAL CON PROTECCIÓN
# ===============================
try:
    for step, row in df.iterrows():

        vehID = f"veh_{row['Veh_ID']}"

        lat, lon = row["Latitud"], row["Longitud"]
        speed_real = row["Velocidad (Km/h)"]
        speed_pred = row["Vel_Pred_kmh"]
        speed_used = speed_pred if not pd.isna(speed_pred) else speed_real

        # GPS → SUMO
        x, y = net.convertLonLat2XY(lon, lat)

        # ===============================
        # ASEGURAR EXISTENCIA
        # ===============================
        if vehID not in traci.vehicle.getIDList():
            traci.vehicle.add(
                vehID=vehID,
                routeID="r0",
                typeID="DEFAULT_VEHTYPE",
                depart=traci.simulation.getTime()
            )
            traci.simulationStep()

        # ===============================
        # BUSCAR EDGE CERCANO
        # ===============================
        edges_near = net.getNeighboringEdges(x, y, 500)

        if not edges_near:
            print(f"Punto fuera de red: {vehID} | step {step}")
            traci.simulationStep()
            continue

        edge_id = edges_near[0][0].getID()

        # ===============================
        # MOVER VEHÍCULO (SEGURO)
        # ===============================
        try:
            traci.vehicle.moveToXY(
                vehID,
                edgeID=edge_id,
                laneIndex=0,
                x=x,
                y=y,
                keepRoute=0,
                matchThreshold=500
            )
        except traci.TraCIException:
            print(f" SUMO no pudo mapear {vehID} | step {step}")
            traci.simulationStep()
            continue

        traci.vehicle.setSpeed(
            vehID,
            max(speed_used, 0) / 3.6
        )

        traci.simulationStep()

        # ===============================
        # REGISTRAR RESULTADOS
        # ===============================
        speed_sumo = traci.vehicle.getSpeed(vehID) * 3.6
        pos_x, pos_y = traci.vehicle.getPosition(vehID)
        co2 = traci.vehicle.getCO2Emission(vehID)

        sumo_results.append([
            step,
            vehID,
            lat,
            lon,
            speed_sumo,
            speed_used,
            speed_real,
            speed_used - speed_real,
            co2,
            pos_x,
            pos_y
        ])

except Exception as e:
    print(f" Error durante la simulación: {e}")

finally:
    # ===============================
    # 6. CERRAR SUMO Y GUARDAR RESULTADOS
    # ===============================
    try:
        traci.close()
    except Exception:
        pass  # Si SUMO ya está cerrado o falla, seguimos

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
            "Y"
        ]
    )

sumo_df.to_csv("resultados_sumo_final.csv", index=False)
print("Simulación finalizada (parcial o completa)")
print("Archivo generado: resultados_sumo_final.csv")
