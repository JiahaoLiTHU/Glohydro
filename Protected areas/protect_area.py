# -*- coding: utf-8 -*-
"""
New Logic (No River Plotting):
1) First retrieve "rivers with hydropower stations" (intersection between hydropower buffer and rivers)
2) Expand these rivers to "entire rivers" (if MAIN_RIV field exists)
3) Use "entire river buffer" to retrieve intersecting protected areas (affected protected areas)
4) Print statistical results + plot (only show protected areas and hydropower stations: Unaffected = light green; Affected = light blue; Hydropower stations = Morandi light purple)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import psutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
from shapely.geometry import Point, box
from shapely.ops import unary_union

# ------------------------------------------------------------------------------
# 1. Local Configuration (Key Paths)
# ------------------------------------------------------------------------------
ROOT_FOLDER = r""  # Root folder for protected area SHP files
CSV_PATH = r""  # Hydropower station CSV path
RIVER_SHP_PATH = r"./HydroRIVERS_v10_shp/HydroRIVERS_v10_shp/HydroRIVERS_v10.shp"  # River SHP path
OUTPUT_IMAGE = r"Protected_Area_River_Impact.png"  # Output image path
LAND_SHP_PATH = r"ne_110m_land.shp"  # Land boundary SHP path

# Latitude/Longitude column names (match CSV)
LONGITUDE_COLUMN_NAME = 'Longitude'
LATITUDE_COLUMN_NAME = 'Latitude'

# Analysis extent (can set bbox, e.g., China [73, 18, 135, 53])
TARGET_EXTENT = None  # Or [73, 18, 135, 53]

# ------------------------------------------------------------------------------
# 2. Parameters (Buffer/Simplification/Display)
# ------------------------------------------------------------------------------
# Simplification tolerance (degrees)
SIMPLIFY_TOLERANCE = 0.01     # For protected areas
RIVER_SIMPLIFY_TOLERANCE = 0.02

# Core buffer distances (degrees)
HYDRO_RIVER_BUFFER = 0.05      # Hydropower -> River matching buffer (~5.5km)
RIVER_PROTECTED_BUFFER = 0.02  # River -> Protected area impact buffer (~2.2km)

# Visual color scheme (per new requirements)
UNAFFECTED_COLOR = "#d9f2d9"   # Unaffected protected areas: light green
AFFECTED_COLOR   = "#a8d8f0"   # Affected protected areas: light blue
HYDRO_MARKER_COLOR = "#c8b6ff" # Hydropower stations: Morandi light purple
HYDRO_MARKER_SIZE  = 10
HYDRO_MARKER_ALPHA = 0
PROTECTED_ALPHA    = 0.9

# Filter extremely short river segments (approximate km measurement)
MIN_RIVER_LEN_KM = 0.5  # Keep only rivers ≥0.5 km

# ------------------------------------------------------------------------------
# 3. Utility Functions
# ------------------------------------------------------------------------------
def print_memory_usage(msg=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    print(f"[Memory] {msg}: {mem:.2f} MB")

def load_land_boundary(land_shp_path):
    print(f"Loading land boundary: {land_shp_path} ...")
    try:
        land_gdf = gpd.read_file(land_shp_path).to_crs(4326)
        return unary_union(land_gdf.geometry)
    except Exception as e:
        print(f"  Failed to load land boundary: {e}, all protected areas treated as land by default")
        return None

def is_land_polygon(poly, land_boundary):
    if land_boundary is None:
        return True
    return poly.centroid.buffer(0.005).intersects(land_boundary)

def load_preprocess_protected_areas(root_folder, land_boundary, target_extent=None, simplify_tol=0.01):
    print(f"Loading protected area data (simplification: {simplify_tol}°) ...")
    target_crs = 4326
    all_parts = []
    for root, _, files in os.walk(root_folder):
        for f in files:
            if f.lower().endswith(".shp") and not f.startswith("~$"):
                shp = os.path.join(root, f)
                try:
                    bbox = box(*target_extent) if target_extent else None
                    gdf = gpd.read_file(shp, bbox=bbox)
                    if gdf.crs is None:
                        gdf.set_crs(target_crs, inplace=True)
                    else:
                        gdf = gdf.to_crs(target_crs)
                    gdf = gdf[gdf.is_valid]
                    if not gdf.empty:
                        gdf = gdf[['geometry']].copy()
                        gdf['geometry'] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)
                        gdf['IS_LAND'] = gdf['geometry'].apply(lambda x: is_land_polygon(x, land_boundary))
                        all_parts.append(gdf)
                    print(f"  Loaded: {shp} ({len(gdf)} features)")
                except Exception as e:
                    print(f"  Failed: {shp} | {e}")
    if not all_parts:
        raise ValueError("No protected area data loaded")
    merged = gpd.GeoDataFrame(pd.concat(all_parts, ignore_index=True), crs=4326)
    merged['PROTECTED_ID'] = np.arange(len(merged))
    merged['IS_RIVER_IMPACTED'] = False
    print(f"Protected area loading completed: {len(merged)} features\n")
    return merged

def load_preprocess_rivers(river_shp_path, target_extent=None, simplify_tol=0.02):
    print(f"Loading river data (simplification: {simplify_tol}°) ...")
    bbox = box(*target_extent) if target_extent else None
    rivers = gpd.read_file(river_shp_path, bbox=bbox)
    if rivers.crs is None:
        rivers.set_crs(4326, inplace=True)
    else:
        rivers = rivers.to_crs(4326)
    rivers = rivers[rivers.is_valid].copy()
    keep_cols = [c for c in ["MAIN_RIV", "HYRIV_ID", "ORD_FLOW", "DIS_AV_CMS"] if c in rivers.columns]
    rivers = rivers[keep_cols + ["geometry"]] if keep_cols else rivers[["geometry"]].copy()
    rivers["geometry"] = rivers.geometry.simplify(simplify_tol, preserve_topology=True)
    rivers["__LEN_KM"] = rivers.geometry.length * 111.0
    rivers = rivers[rivers["__LEN_KM"] >= MIN_RIVER_LEN_KM].copy()
    if "HYRIV_ID" in rivers.columns:
        rivers["RIVER_ID"] = rivers["HYRIV_ID"]
    else:
        rivers["RIVER_ID"] = np.arange(len(rivers))
    print(f"River loading completed: {len(rivers)} segments (≥{MIN_RIVER_LEN_KM}km)\n")
    return rivers

def load_hydro_stations(csv_path, target_extent=None):
    print(f"Loading hydropower station data: {csv_path} ...")
    df = pd.read_csv(csv_path)
    df[LONGITUDE_COLUMN_NAME] = pd.to_numeric(df[LONGITUDE_COLUMN_NAME], errors="coerce")
    df[LATITUDE_COLUMN_NAME] = pd.to_numeric(df[LATITUDE_COLUMN_NAME], errors="coerce")
    df.dropna(subset=[LONGITUDE_COLUMN_NAME, LATITUDE_COLUMN_NAME], inplace=True)
    if target_extent:
        minx, miny, maxx, maxy = target_extent
        pad = 0.1
        df = df[(df[LONGITUDE_COLUMN_NAME] >= minx - pad) &
                (df[LONGITUDE_COLUMN_NAME] <= maxx + pad) &
                (df[LATITUDE_COLUMN_NAME]  >= miny - pad) &
                (df[LATITUDE_COLUMN_NAME]  <= maxy + pad)]
    geom = [Point(xy) for xy in zip(df[LONGITUDE_COLUMN_NAME], df[LATITUDE_COLUMN_NAME])]
    hyd = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
    hyd = hyd.drop_duplicates(subset=[LONGITUDE_COLUMN_NAME, LATITUDE_COLUMN_NAME])
    print(f"Hydropower station loading completed: {len(hyd)} stations\n")
    return hyd

# ------------------------------------------------------------------------------
# 4) First find "rivers with hydropower stations", then expand to "entire rivers"
# ------------------------------------------------------------------------------
def find_rivers_with_hydro(hydro_gdf: gpd.GeoDataFrame,
                           river_gdf: gpd.GeoDataFrame,
                           buffer_distance_deg: float = HYDRO_RIVER_BUFFER) -> gpd.GeoDataFrame:
    """
    Use sjoin to find intersections between "hydropower station buffers" and rivers,
    resulting in "rivers with hydropower stations".
    If MAIN_RIV field exists, expand to entire rivers (all segments with the same MAIN_RIV)
    """
    print("Retrieving rivers with hydropower stations ...")
    buf = hydro_gdf.copy()
    buf["geometry"] = buf.geometry.buffer(buffer_distance_deg)
    hits = gpd.sjoin(river_gdf, buf[["geometry"]], how="inner", predicate="intersects")
    if hits.empty:
        print("  No 'hydropower-river' intersection found, attempting to double buffer distance ...")
        buf["geometry"] = hydro_gdf.geometry.buffer(buffer_distance_deg * 2)
        hits = gpd.sjoin(river_gdf, buf[["geometry"]], how="inner", predicate="intersects")
    if hits.empty:
        raise ValueError("Still no rivers with hydropower stations found. Please check coordinates/extent/buffer parameters.")

    rivers_hit = river_gdf.loc[hits.index.unique()].copy()
    print(f"  Number of hit river segments: {len(rivers_hit)}")

    if "MAIN_RIV" in river_gdf.columns:
        mains = rivers_hit["MAIN_RIV"].dropna().unique().tolist()
        rivers_full = river_gdf[river_gdf["MAIN_RIV"].isin(mains)].copy()
        print(f"  Expanded to entire rivers (via MAIN_RIV): {len(set(mains))} unique rivers; Total segments: {len(rivers_full)}")
    else:
        rivers_full = rivers_hit
        print("  Note: MAIN_RIV field not found, cannot expand to entire rivers. Keeping only hit segments.")

    return rivers_full

# ------------------------------------------------------------------------------
# 5) Retrieve affected protected areas using "entire river buffers"
# ------------------------------------------------------------------------------
def find_impacted_protected_by_whole_rivers(rivers_full: gpd.GeoDataFrame,
                                            protected_gdf: gpd.GeoDataFrame,
                                            buffer_distance_deg: float = RIVER_PROTECTED_BUFFER
                                            ) -> gpd.GeoDataFrame:
    """
    Buffer entire rivers (lines) and intersect with protected areas (polygons)
    to get affected protected areas
    """
    print("Retrieving protected areas affected by entire rivers ...")
    river_buf = rivers_full.copy()
    river_buf["geometry"] = river_buf.geometry.buffer(buffer_distance_deg)
    join = gpd.sjoin(protected_gdf[["PROTECTED_ID", "IS_LAND", "geometry"]],
                     river_buf[["geometry"]],
                     how="left", predicate="intersects")
    impacted_ids = join.loc[join.index_right.notna(), "PROTECTED_ID"].unique().tolist()
    protected_gdf = protected_gdf.copy()
    protected_gdf["IS_RIVER_IMPACTED"] = protected_gdf["PROTECTED_ID"].isin(impacted_ids)
    print(f"  Number of affected protected areas: {protected_gdf['IS_RIVER_IMPACTED'].sum()}")
    return protected_gdf

# ------------------------------------------------------------------------------
# 6) Plotting (No rivers; only protected areas and hydropower stations)
# ------------------------------------------------------------------------------
def plot_result_map(protected_gdf, hydro_gdf):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(22, 12))

    impacted = protected_gdf[protected_gdf["IS_RIVER_IMPACTED"]]
    noimpact = protected_gdf[~protected_gdf["IS_RIVER_IMPACTED"]]

    # 1) Unaffected protected areas: light green
    if not noimpact.empty:
        noimpact.plot(ax=ax, color=UNAFFECTED_COLOR, alpha=PROTECTED_ALPHA, edgecolor='none',
                      label=f"Unaffected Protected Areas: {len(noimpact)}")

    # 2) Affected protected areas: light blue (covers entire area)
    if not impacted.empty:
        impacted.plot(ax=ax, color=AFFECTED_COLOR, alpha=PROTECTED_ALPHA, edgecolor='none',
                      label=f"Affected Protected Areas: {len(impacted)}")

    # 3) Hydropower stations: Morandi light purple
    if not hydro_gdf.empty:
        ax.scatter(hydro_gdf[LONGITUDE_COLUMN_NAME], hydro_gdf[LATITUDE_COLUMN_NAME],
                   s=HYDRO_MARKER_SIZE, c=HYDRO_MARKER_COLOR, alpha=HYDRO_MARKER_ALPHA,
                   label=f"Hydropower Stations: {len(hydro_gdf)}")

    # Set view extent
    if TARGET_EXTENT:
        ax.set_xlim(TARGET_EXTENT[0], TARGET_EXTENT[2])
        ax.set_ylim(TARGET_EXTENT[1], TARGET_EXTENT[3])
    else:
        bounds = protected_gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

    #ax.set_title("Protected Areas Affected by Entire Rivers (Blue) vs Unaffected (Green); Hydropower Stations as Light Purple Dots", fontsize=15, fontweight='bold', pad=18)
    #ax.set_xlabel("Longitude (°E)")
    #ax.set_ylabel("Latitude (°N)")
    #ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    #ax.legend(loc="lower left", fontsize=10, frameon=True, ncol=2)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Map saved to: {OUTPUT_IMAGE}")

# ------------------------------------------------------------------------------
# 7) Statistical Output (Independent of plotting)
# ------------------------------------------------------------------------------
def print_statistics(hydro_gdf, rivers_full, protected_gdf):
    impacted = protected_gdf[protected_gdf["IS_RIVER_IMPACTED"]]
    total_pa = len(protected_gdf)
    total_imp = len(impacted)
    if "IS_LAND" in protected_gdf.columns:
        imp_land  = int(impacted["IS_LAND"].sum())
        imp_ocean = int(total_imp - imp_land)
    else:
        imp_land, imp_ocean = np.nan, np.nan

    # Deduplicated count of entire rivers
    if "MAIN_RIV" in rivers_full.columns:
        n_main = rivers_full["MAIN_RIV"].nunique()
    else:
        n_main = rivers_full["RIVER_ID"].nunique()

    print("\n====== Statistical Results ======")
    print(f"Number of hydropower stations: {len(hydro_gdf)}")
    print(f"Unique entire rivers with hydropower stations: {n_main}")
    print(f"Total number of protected areas: {total_pa}")
    print(f"Total affected protected areas: {total_imp}" + (f"  (Land: {imp_land}, Ocean: {imp_ocean})" if not np.isnan(imp_land) else ""))
    if total_pa > 0:
        print(f"Affected ratio: {total_imp/total_pa*100:.2f}%")

# ------------------------------------------------------------------------------
# 8) Main Workflow
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("="*60)
        print("Step 1: Load Base Data")
        print("="*60)
        land_boundary = load_land_boundary(LAND_SHP_PATH)
        protected_gdf = load_preprocess_protected_areas(
            ROOT_FOLDER, land_boundary, TARGET_EXTENT, SIMPLIFY_TOLERANCE
        )
        river_gdf = load_preprocess_rivers(
            RIVER_SHP_PATH, TARGET_EXTENT, RIVER_SIMPLIFY_TOLERANCE
        )
        hydro_gdf = load_hydro_stations(CSV_PATH, TARGET_EXTENT)
        print_memory_usage("Data loading completed")

        print("\n" + "="*60)
        print("Step 2: Retrieve Rivers with Hydropower -> Expand to Entire Rivers")
        print("="*60)
        rivers_with_hydro_full = find_rivers_with_hydro(hydro_gdf, river_gdf, HYDRO_RIVER_BUFFER)
        print_memory_usage("Retrieval-Expansion completed")

        print("\n" + "="*60)
        print("Step 3: Retrieve Affected Protected Areas (Entire River Buffers)")
        print("="*60)
        protected_gdf = find_impacted_protected_by_whole_rivers(
            rivers_with_hydro_full, protected_gdf, RIVER_PROTECTED_BUFFER
        )
        print_memory_usage("Protected area retrieval completed")

        print("\n" + "="*60)
        print("Step 4: Output Statistics and Plot (No River Plotting)")
        print("="*60)
        print_statistics(hydro_gdf, rivers_with_hydro_full, protected_gdf)
        plot_result_map(protected_gdf, hydro_gdf)

        print("\n✅ Completed.")
    except Exception as e:
        print(f"\nProgram Error: {e}")
        import traceback; traceback.print_exc()