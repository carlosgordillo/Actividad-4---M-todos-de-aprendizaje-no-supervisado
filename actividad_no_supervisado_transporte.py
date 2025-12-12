# ================================================
#   APRENDIZAJE NO SUPERVISADO EN TRANSPORTE MASIVO
#   K-MEANS + DBSCAN
#   Dataset Sintético de Paraderos
# ================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ---------------------------
# 1. CREACIÓN DEL DATASET
# ---------------------------

data = {
    "stop_id": range(1, 16),
    "stop_name": [
        "Central", "Terminal Norte", "La Esperanza", "Industrial",
        "Rural 1", "Rural 2", "Comercial", "Hospital", "Colegio",
        "Zona Franca", "Altos", "Centro 2", "Mercado", "Universidad", "Parque"
    ],
    "lat": [5.61, 5.62, 5.63, 5.57, 5.70, 5.71, 5.60, 5.58, 5.59, 5.61, 5.73, 5.64, 5.62, 5.63, 5.65],
    "lon": [-73.52, -73.53, -73.50, -73.48, -73.60, -73.61, -73.49, -73.47, -73.46, -73.50, -73.62, -73.51, -73.48, -73.47, -73.49],
    "avg_passengers_per_hour": [120, 150, 90, 70, 20, 15, 110, 80, 60, 95, 10, 85, 130, 140, 100],
    "dwell_time_sec": [50, 60, 40, 35, 20, 15, 55, 45, 38, 48, 10, 42, 58, 65, 50],
    "boarding_type": ["frontal", "medio", "trasero", "frontal", "frontal", "medio", "trasero",
                      "frontal", "medio", "trasero", "frontal", "medio", "trasero", "frontal", "medio"],
    "fare_cop": [3000, 3000, 2800, 2800, 2600, 2600, 3000, 2800, 2700, 2900, 2500, 2900, 3000, 3000, 2800]
}

df = pd.DataFrame(data)

# Codificación del tipo de embarque
df["boarding_type_code"] = df["boarding_type"].map({"frontal": 0, "medio": 1, "trasero": 2})

# Nueva variable
df["pass_per_dwell"] = df["avg_passengers_per_hour"] / df["dwell_time_sec"]

print("Dataset cargado correctamente\n")
print(df.head())


# ---------------------------
# 2. PREPROCESAMIENTO
# ---------------------------

features = df[[
    "lat", "lon", "avg_passengers_per_hour",
    "dwell_time_sec", "boarding_type_code", "pass_per_dwell"
]]

scaler = StandardScaler()
X = scaler.fit_transform(features)


# ---------------------------
# 3. K-MEANS (2 a 5 clusters)
# ---------------------------

best_k = 0
best_score = -1
best_labels = None

for k in range(2, 6):
    model_k = KMeans(n_clusters=k, n_init=10)
    labels = model_k.fit_predict(X)
    score = silhouette_score(X, labels)

    print(f"Silhouette Score para k={k}: {score}")

    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

df["kmeans_cluster"] = best_labels

print(f"\nMejor valor de k: {best_k} con Silhouette Score = {best_score}\n")


# ---------------------------
# 4. MODELO DBSCAN
# ---------------------------

dbscan = DBSCAN(eps=0.9, min_samples=2)
dbscan_labels = dbscan.fit_predict(X)

df["dbscan_cluster"] = dbscan_labels

print("Clusters DBSCAN asignados:")
print(dbscan_labels)


# ---------------------------
# 5. VISUALIZACIONES
# ---------------------------

plt.figure(figsize=(8, 6))
plt.scatter(df["lat"], df["lon"], c=df["kmeans_cluster"], cmap="viridis")
plt.title("Clustering K-Means (Coordenadas)")
plt.xlabel("Latitud")
plt.ylabel("Longitud")
plt.colorbar(label="Cluster")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df["lat"], df["lon"], c=df["dbscan_cluster"], cmap="plasma")
plt.title("Clustering DBSCAN (Coordenadas)")
plt.xlabel("Latitud")
plt.ylabel("Longitud")
plt.colorbar(label="Cluster")
plt.show()


# ---------------------------
# 6. EXPORTAR RESULTADOS
# ---------------------------

df.to_csv("stops_with_clusters.csv", index=False)

summary = df.groupby("kmeans_cluster")[
    ["avg_passengers_per_hour", "dwell_time_sec", "pass_per_dwell"]
].mean()

summary.to_csv("kmeans_summary.csv")

print("\nArchivos generados:")
print(" - stops_with_clusters.csv")
print(" - kmeans_summary.csv")

print("\nEjecución finalizada correctamente.")
