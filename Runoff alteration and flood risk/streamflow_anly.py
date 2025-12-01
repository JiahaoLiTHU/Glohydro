# ============================================================
#   Streamflow Analysis – Plants + Nearby Rivers (Hydrology)
#   Output daily/monthly/yearly flow, trends, dashboard,
#   and knowledge graph files.
# ============================================================

import os
import numpy as np
import pandas as pd
import xarray as xr

from typing import Iterable, Optional, Tuple, List, Dict, Union

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# -------------------------------
# Paths (Modify These)
# -------------------------------
base_directory = './data/hydrologic_predictions/model_id_8583a5c2_v0/'
PATH_REANALYSIS = os.path.join(base_directory, 'reanalysis/streamflow.zarr/')
PATH_REFORECAST = os.path.join(base_directory, 'reforecast/streamflow.zarr/')
PATH_OUTLETS    = os.path.join(base_directory, 'hybas_outlet_locations_UNOFFICIAL.zarr/')

# ============================================================
# Dataset Loading
# ============================================================

def load_datasets(
    path_reanalysis: str = PATH_REANALYSIS,
    path_outlets: str = PATH_OUTLETS,
    path_reforecast: Optional[str] = PATH_REFORECAST
) -> Tuple[xr.Dataset, xr.Dataset, Optional[xr.Dataset]]:
    reanalysis_ds = xr.open_zarr(path_reanalysis, consolidated=False)
    outlets_ds   = xr.open_zarr(path_outlets, consolidated=False)
    reforecast_ds = None
    if path_reforecast:
        try:
            reforecast_ds = xr.open_zarr(path_reforecast, consolidated=False)
        except Exception:
            reforecast_ds = None
    return reanalysis_ds, outlets_ds, reforecast_ds

# ============================================================
# Utility Functions
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def normalize_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180) % 360) - 180

# ============================================================
# Gauge Finding
# ============================================================

def find_nearest_gauge(outlets_ds: xr.Dataset, lat: float, lon: float):
    lats = outlets_ds["latitude"].values
    lons = normalize_lon(outlets_ds["longitude"].values)
    lon  = float(normalize_lon(np.array([lon]))[0])
    dists = haversine_km(lat, lon, lats, lons)
    idx = int(np.argmin(dists))
    gauge_id = outlets_ds["gauge_id"].values[idx].item()
    return gauge_id, float(lats[idx]), float(lons[idx]), float(dists[idx])

def find_gauges_within_radius(outlets_ds, lat, lon, radius_km, max_results=None):
    lats = outlets_ds["latitude"].values
    lons = normalize_lon(outlets_ds["longitude"].values)
    lon  = float(normalize_lon(np.array([lon]))[0])
    dists = haversine_km(lat, lon, lats, lons)
    mask = dists <= radius_km
    df = pd.DataFrame({
        "gauge_id": outlets_ds["gauge_id"].values[mask].astype(str),
        "latitude": lats[mask],
        "longitude": lons[mask],
        "distance_km": dists[mask]
    }).sort_values("distance_km").reset_index(drop=True)
    if max_results:
        df = df.iloc[:max_results]
    return df

# ============================================================
# Timeseries Extraction
# ============================================================

def get_reanalysis_timeseries(reanalysis_ds, gauge_ids):
    sel_ids = [str(g) for g in list(gauge_ids)]
    subset = reanalysis_ds.sel(gauge_id=sel_ids)

    cand_vars = [v for v in subset.data_vars if v.lower() in ("discharge","streamflow","flow","q")]
    if not cand_vars:
        cand_vars = [v for v in subset.data_vars if np.issubdtype(subset[v].dtype, np.number)]

    return subset[cand_vars]

# ============================================================
# Single-Point Query (Reusable)
# ============================================================

# def query_runoff_by_point(
#     lat: float,
#     lon: float,
#     radius_km: float = 25.0,
#     max_results: int = 5,
#     reanalysis_ds: Optional[xr.Dataset] = None,
#     outlets_ds: Optional[xr.Dataset] = None,
#     reforecast_ds: Optional[xr.Dataset] = None,
# ) -> Dict[str, Union[xr.Dataset, pd.DataFrame, None]]:

#     if reanalysis_ds is None or outlets_ds is None:
#         reanalysis_ds, outlets_ds, rf_ds = load_datasets()
#         if reforecast_ds is None:
#             reforecast_ds = rf_ds

#     nearby = find_gauges_within_radius(outlets_ds, lat, lon, radius_km, max_results=max_results)

#     # if nearby.empty:
#     #     gid, glat, glon, dkm = find_nearest_gauge(outlets_ds, lat, lon)
#     #     nearby = pd.DataFrame([{
#     #         "gauge_id": gid,
#     #         "latitude": glat,
#     #         "longitude": glon,
#     #         "distance_km": dkm
#     #     }])
#     # === 新增：如果根本没有 gauge，直接返回空，不继续处理 ===
#     if nearby.empty:
#         try:
#             gid, glat, glon, dkm = find_nearest_gauge(outlets_ds, lat, lon)
#             nearby = pd.DataFrame([{
#                 "gauge_id": gid,
#                 "latitude": glat,
#                 "longitude": glon,
#                 "distance_km": dkm
#             }])
#         except Exception:
#             print(f"⚠ WARNING: no gauge found near Plant at (lat={lat}, lon={lon}). Skipping.")
#             return {"nearby": pd.DataFrame(), "reanalysis": None}
        
#     rean = get_reanalysis_timeseries(reanalysis_ds, nearby["gauge_id"].tolist())
#     return {"nearby": nearby, "reanalysis": rean}

def query_runoff_by_point(
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    max_results: int = 1,
    reanalysis_ds: Optional[xr.Dataset] = None,
    outlets_ds: Optional[xr.Dataset] = None,
    reforecast_ds: Optional[xr.Dataset] = None,
) -> Dict[str, Union[xr.Dataset, pd.DataFrame, None]]:

    if reanalysis_ds is None or outlets_ds is None:
        reanalysis_ds, outlets_ds, rf_ds = load_datasets()
        if reforecast_ds is None:
            reforecast_ds = rf_ds

    # -------- Auto-expand radius search --------
    search_radii = [radius_km, 30, 50, 100, 200, 500, 1000]

    nearby = pd.DataFrame()

    for r in search_radii:
        nearby = find_gauges_within_radius(
            outlets_ds, lat, lon, radius_km=r, max_results=max_results
        )
        if not nearby.empty:
            print(f"✓ Found gauge within {r} km.")
            break

    # If still empty → fallback to global nearest gauge
    if nearby.empty:
        try:
            gid, glat, glon, dkm = find_nearest_gauge(outlets_ds, lat, lon)
            nearby = pd.DataFrame([{
                "gauge_id": gid,
                "latitude": glat,
                "longitude": glon,
                "distance_km": dkm
            }])
            print(f"⚠ No gauge found within search radii; using global nearest gauge at {dkm:.1f} km.")
        except Exception:
            print(f"🚫 ERROR: No usable gauge found globally near lat={lat}, lon={lon}")
            return {"nearby": pd.DataFrame(), "reanalysis": None}

    # Extract time series
    try:
        rean = get_reanalysis_timeseries(reanalysis_ds, nearby["gauge_id"].tolist())
    except Exception:
        print(f"⚠ Reanalysis missing for selected gauge(s). Skipping.")
        return {"nearby": nearby, "reanalysis": None}

    return {"nearby": nearby, "reanalysis": rean}


# ============================================================
# Main Processing
# ============================================================

def process_plants_from_csv(
    plant_csv_path: str,
    radius_km: float = 10.0,
    max_rivers: int = 1,
    output_root: str = "./output_plants",
    reanalysis_ds: Optional[xr.Dataset] = None,
    outlets_ds: Optional[xr.Dataset] = None,
):

    if reanalysis_ds is None or outlets_ds is None:
        reanalysis_ds, outlets_ds, _ = load_datasets()

    plants_df = pd.read_csv(plant_csv_path)
    # plants_df = plants_df.head(10)     # Only process first 10 for now

    os.makedirs(output_root, exist_ok=True)

    summary_records = []
    node_records = {}
    edge_records = []

    for _, row in plants_df.iterrows():
        plant_id = row["PlantID"]
        plat = float(row["Latitude"])
        plon = float(row["Longitude"])

        print(f"=== Processing Plant {plant_id} ===")

        plant_dir = os.path.join(output_root, f"plant_{plant_id}")
        os.makedirs(plant_dir, exist_ok=True)

        # === 如果电站已经处理过（有 dashboard 文件），则跳过 ===
        dashboard_file = os.path.join(plant_dir, f"plant_{plant_id}_dashboard.html")
        if os.path.exists(dashboard_file):
            print(f"⏩ Skipping Plant {plant_id}: already processed.")
            continue


        result = query_runoff_by_point(
            plat, plon, radius_km=radius_km, max_results=max_rivers,
            reanalysis_ds=reanalysis_ds, outlets_ds=outlets_ds
        )

        nearby = result["nearby"]
        rean_ds = result["reanalysis"]


        # === 关键安全检查（避免 rean_ds=None 崩溃） ===
        if nearby.empty or (rean_ds is None) or (len(rean_ds.data_vars) == 0):
            print(f"🚫 Skipping plant {plant_id}: no usable gauge or no streamflow data.")
            continue

        flow_var = list(rean_ds.data_vars)[0]

        # Node: plant
        plant_node_id = f"plant_{plant_id}"
        node_records[plant_node_id] = {
            "node_id": plant_node_id,
            "node_type": "plant",
            "label": str(plant_id),
            "latitude": plat,
            "longitude": plon,
        }

        # ======================================================
        # Process Each Gauge
        # ======================================================
        for _, g in nearby.iterrows():
            gauge_id = g["gauge_id"]
            glat = g["latitude"]
            glon = g["longitude"]
            dist = g["distance_km"]

            ds_one = rean_ds.sel(gauge_id=gauge_id)
            ts = ds_one[flow_var].to_pandas()
            ts.name = "flow"

            # Monthly and Yearly
            ts_monthly = ts.resample("M").mean()
            ts_yearly  = ts.resample("Y").mean()

            # Save daily/monthly/yearly
            ts.to_frame().to_csv(os.path.join(plant_dir, f"gauge_{gauge_id}_daily.csv"))
            ts_monthly.to_frame().to_csv(os.path.join(plant_dir, f"gauge_{gauge_id}_monthly.csv"))
            ts_yearly.to_frame().to_csv(os.path.join(plant_dir, f"gauge_{gauge_id}_yearly.csv"))

            # Stats
            total_flow = ts.sum()
            mean_flow  = ts.mean()
            yearly_mean = ts_yearly.mean()
            monthly_mean = ts_monthly.mean()

            summary_records.append({
                "PlantID": plant_id,
                "GaugeID": gauge_id,
                "Distance_km": dist,
                "TotalFlow": total_flow,
                "MeanFlow_daily": mean_flow,
                "MeanFlow_monthly": float(monthly_mean),
                "MeanFlow_yearly": float(yearly_mean),
            })

            # Node: gauge
            gauge_node_id = f"gauge_{gauge_id}"
            node_records[gauge_node_id] = {
                "node_id": gauge_node_id,
                "node_type": "gauge",
                "label": gauge_id,
                "latitude": glat,
                "longitude": glon,
            }

            # Edge
            edge_records.append({
                "from_id": plant_node_id,
                "to_id": gauge_node_id,
                "distance_km": dist,
                "total_flow": total_flow
            })

        # ======================================================
        # Seaborn Plot: Monthly
        # ======================================================
        fig, ax = plt.subplots(figsize=(12,5))
        for g_id in nearby["gauge_id"]:
            df = pd.read_csv(os.path.join(plant_dir, f"gauge_{g_id}_monthly.csv"))
            df["time"] = pd.to_datetime(df["time"])
            sns.lineplot(data=df, x="time", y="flow", ax=ax, label=f"Gauge {g_id}")

        plt.title(f"Plant {plant_id} – Monthly Mean Streamflow")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(plant_dir, f"plant_{plant_id}_monthly_trend.png"))
        plt.close()

        # ======================================================
        # Seaborn Plot: Yearly
        # ======================================================
        fig, ax = plt.subplots(figsize=(12,5))
        for g_id in nearby["gauge_id"]:
            df = pd.read_csv(os.path.join(plant_dir, f"gauge_{g_id}_yearly.csv"))
            df["time"] = pd.to_datetime(df["time"])
            sns.lineplot(data=df, x="time", y="flow", marker="o", ax=ax, label=f"Gauge {g_id}")

        plt.title(f"Plant {plant_id} – Yearly Mean Streamflow")
        plt.tight_layout()
        plt.savefig(os.path.join(plant_dir, f"plant_{plant_id}_yearly_trend.png"))
        plt.close()

        # ======================================================
        # HTML Dashboard
        # ======================================================
        summary_df_local = pd.DataFrame(summary_records)
        summary_df_local = summary_df_local[summary_df_local["PlantID"] == plant_id]

        dashboard_html = os.path.join(plant_dir, f"plant_{plant_id}_dashboard.html")

        html_content = f"""
        <html>
        <head>
            <title>Plant {plant_id} Dashboard</title>
            <style>
                body {{ font-family: Arial; margin: 30px; }}
                img {{ max-width: 100%; border: 1px solid #ccc; }}
                h1, h2 {{ text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Plant {plant_id} – Streamflow Dashboard</h1>

            <h2>Monthly Mean Trend</h2>
            <img src="plant_{plant_id}_monthly_trend.png">

            <h2>Yearly Mean Trend</h2>
            <img src="plant_{plant_id}_yearly_trend.png">

            <h2>Summary Table</h2>
            {summary_df_local.to_html(index=False)}
        </body>
        </html>
        """

        with open(dashboard_html, "w") as f:
            f.write(html_content)

    # ======================================================
    # Global Summary
    # ======================================================
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(output_root, "plant_river_summary.csv"), index=False)

    pd.DataFrame(node_records.values()).to_csv(
        os.path.join(output_root, "knowledge_graph_nodes.csv"), index=False)
    pd.DataFrame(edge_records).to_csv(
        os.path.join(output_root, "knowledge_graph_edges.csv"), index=False)

    return summary_df

# ======================================================
# Run (Edit plant_csv_path to your file)
# ======================================================

if __name__ == "__main__":
    summary = process_plants_from_csv(
        plant_csv_path="./data/power_station_locations/hydro_final_new_.csv",
        radius_km=10.0,
        max_rivers=1,
        output_root="./output_plants"
    )

    print(summary.head())
