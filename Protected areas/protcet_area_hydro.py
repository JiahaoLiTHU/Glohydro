import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import psutil  # New: For memory monitoring
from pyproj import CRS
from shapely.geometry import Point, box

# ------------------------------------------------------------------------------
# 1. Local Configuration (Please modify these paths)
# ------------------------------------------------------------------------------
ROOT_FOLDER = r"./world"  # Root folder for SHP files
CSV_PATH = r"./Figure/1/unmatched.csv"  # Path to your CSV file
OUTPUT_IMAGE = r"./world/Protected_Area_Heatmap_Optimized_Simplified.png"  # Output image path

# Column names for latitude and longitude in the CSV file - modify according to your actual data
LONGITUDE_COLUMN_NAME = 'Longitude'
LATITUDE_COLUMN_NAME = 'Latitude'

# Optional: Define the latitude/longitude extent for analysis and plotting (e.g., [100, 20, 120, 40] = 100-120°E, 20-40°N)
# Set to None to analyze all data. Strongly recommended to set this for significant efficiency gains!
TARGET_EXTENT = None  # Example: [100, 20, 120, 40]

# ------------------------------------------------------------------------------
# 2. Performance and Simplification Parameters (Adjust as needed)
# ------------------------------------------------------------------------------
CSV_CHUNK_SIZE = 50_000  # CSV chunk size. Lower if memory is limited.

# Geometry simplification parameter (unit: degrees). This is the key optimization!
# Larger values = higher simplification, lower memory usage, but less precise boundaries.
# Smaller values = more precise boundaries, but higher memory usage.
# Start with larger values and find a balance between performance and precision.
# For WGS84 coordinate system (EPSG:4326), 0.01 degrees ≈ 1.1 kilometers.
SIMPLIFY_TOLERANCE = 0.01

# ------------------------------------------------------------------------------
# 3. Function Definitions
# ------------------------------------------------------------------------------

def print_memory_usage(message=""):
    """Print current memory usage of the Python process (in MB)"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / (1024 ** 2)  # Convert to megabytes
    print(f"  [Memory Usage] {message}: {mem_mb:.2f} MB")


def load_preprocess_and_simplify_polygons(root_folder, target_extent=None, simplify_tolerance=0.01):
    """
    Load all SHP files, perform preprocessing, clipping, and geometry simplification.
    Returns a generator that yields one processed GeoDataFrame at a time.
    """
    print(f"Starting to read, preprocess, and simplify SHP files in {root_folder} and subfolders...")
    print(f"  - Geometry simplification tolerance: {simplify_tolerance} degrees")
    target_crs = CRS.from_epsg(4326)

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".shp") and not file.startswith("~$"):
                shp_path = os.path.join(root, file)
                try:
                    # Read the file
                    gdf = gpd.read_file(shp_path)

                    # Process name column
                    name_cols = [col for col in gdf.columns if 'name' in col.lower() and col != 'geometry']
                    if 'NAME' not in gdf.columns:
                        if name_cols:
                            gdf.rename(columns={name_cols[0]: 'NAME'}, inplace=True)
                        else:
                            gdf['NAME'] = ""
                    gdf = gdf[['NAME', 'geometry']]

                    # Standardize coordinate system
                    if gdf.crs is None:
                        gdf.set_crs(target_crs, inplace=True)
                    elif gdf.crs != target_crs:
                        gdf = gdf.to_crs(target_crs)

                    # Fix invalid geometries
                    if not gdf.is_valid.all():
                        gdf['geometry'] = gdf.geometry.buffer(0)
                        gdf = gdf[gdf.is_valid]

                    # --- Core optimization: Geometry simplification ---
                    if not gdf.empty:
                        gdf['geometry'] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

                    # Clip to target extent if specified
                    if target_extent:
                        minx, miny, maxx, maxy = target_extent
                        bbox = box(minx, miny, maxx, maxy)
                        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=target_crs)
                        gdf = gpd.clip(gdf, bbox_gdf)

                    if not gdf.empty:
                        print(f"  - Successfully loaded, simplified, and clipped: {shp_path} | Valid records: {len(gdf)}")
                        yield gdf
                    else:
                        print(f"  - File {shp_path} is empty after processing, skipped.")

                except Exception as e:
                    print(f"  - Failed to process {shp_path}: {str(e)} | Skipping this file")

    print("\nAll SHP files have been preprocessed and simplified.")


def count_points_in_polygons_in_chunks(polygon_generator, csv_path, lon_col, lat_col, chunk_size=10_000):
    """Process CSV point data in chunks and perform spatial join analysis with polygon generator."""
    print(f"\nStarting to read CSV point data in chunks and perform spatial analysis: {csv_path}")
    print(f"  - Chunk size: {chunk_size}")
    counts_dict = {}

    for i, points_df_chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        print(f"  - Processing chunk {i + 1}...")
        print_memory_usage("Before processing point chunk")

        # Data cleaning and conversion
        if lon_col not in points_df_chunk.columns or lat_col not in points_df_chunk.columns:
            raise ValueError(f"Error: Specified latitude/longitude columns '{lon_col}' or '{lat_col}' not found in CSV chunk")

        points_df_chunk[lon_col] = pd.to_numeric(points_df_chunk[lon_col], errors='coerce')
        points_df_chunk[lat_col] = pd.to_numeric(points_df_chunk[lat_col], errors='coerce')
        points_df_chunk.dropna(subset=[lon_col, lat_col], inplace=True)

        if points_df_chunk.empty:
            print("    - This chunk is empty or contains invalid coordinates, skipped.")
            continue

        # Create GeoDataFrame for points
        geometry = [Point(xy) for xy in zip(points_df_chunk[lon_col], points_df_chunk[lat_col])]
        points_gdf_chunk = gpd.GeoDataFrame(points_df_chunk, crs=CRS.from_epsg(4326), geometry=geometry)

        print_memory_usage("After creating point GeoDataFrame")

        # For current point chunk, iterate through all polygon files for spatial join
        for polygons_gdf in polygon_generator:
            # Perform spatial join (keep only points within polygons)
            joined_gdf = gpd.sjoin(points_gdf_chunk, polygons_gdf[['NAME', 'geometry']], how="inner",
                                   predicate='within')

            # Update count dictionary
            if not joined_gdf.empty:
                chunk_counts = joined_gdf['NAME'].value_counts()
                for name, count in chunk_counts.items():
                    counts_dict[name] = counts_dict.get(name, 0) + count

        print_memory_usage("After processing one point chunk")

        # Clean up variables to aid garbage collection
        del points_df_chunk, points_gdf_chunk, joined_gdf

    print("\nAll CSV chunks have been analyzed.")
    return counts_dict


def plot_protected_areas_with_heat_from_counts(polygon_generator, counts_dict):
    """Plot heatmap based on count results and simplified polygon data."""
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'SimHei', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    counts_df = pd.DataFrame(list(counts_dict.items()), columns=['NAME', 'POINT_COUNT'])

    all_gdf_to_plot = []
    print("\nPreparing plotting data...")
    for polygons_gdf in polygon_generator:
        merged_gdf = polygons_gdf.merge(counts_df, on='NAME', how='left')
        merged_gdf['POINT_COUNT'] = merged_gdf['POINT_COUNT'].fillna(0).astype(int)
        all_gdf_to_plot.append(merged_gdf)

    if not all_gdf_to_plot:
        print("No polygon data available for plotting.")
        return

    final_plot_gdf = gpd.GeoDataFrame(pd.concat(all_gdf_to_plot, ignore_index=True), crs=all_gdf_to_plot[0].crs)

    gdf_with_points = final_plot_gdf[final_plot_gdf['POINT_COUNT'] > 0]
    gdf_without_points = final_plot_gdf[final_plot_gdf['POINT_COUNT'] == 0]

    if not gdf_without_points.empty:
        gdf_without_points.plot(ax=ax, color="#f0f0f0", edgecolor="none", alpha=0.5)

    if not gdf_with_points.empty:
        gdf_with_points.plot(
            ax=ax,
            column='POINT_COUNT',
            cmap='YlOrRd',
            scheme='quantiles',
            legend=True,
            legend_kwds={
                'loc': 'lower left',
                'title': 'Number of Points in Protected Area',
                'fontsize': 10,
                'title_fontsize': 12
            },
            edgecolor='white',
            linewidth=0.2
        )
    else:
        print("\nWarning: No protected areas contain any points from the CSV file.")
        final_plot_gdf.plot(ax=ax, color="#f0f0f0", edgecolor="none", alpha=0.5)

    if TARGET_EXTENT:
        ax.set_xlim(TARGET_EXTENT[0], TARGET_EXTENT[2])
        ax.set_ylim(TARGET_EXTENT[1], TARGET_EXTENT[3])
    else:
        minx, miny, maxx, maxy = final_plot_gdf.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    total_points_in_areas = final_plot_gdf['POINT_COUNT'].sum()
    ax.set_title(
        f"Protected Area Point Distribution Heatmap (Simplified Version)\n"
        f"CSV File: {os.path.basename(CSV_PATH)} | Total Points: {total_points_in_areas}\n"
        f"Geometry Simplification Tolerance: {SIMPLIFY_TOLERANCE} degrees",
        fontsize=14,
        pad=20
    )
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    print(f"\nHeatmap saved to: {OUTPUT_IMAGE}")
    plt.show()


# ------------------------------------------------------------------------------
# 4. Execute Main Workflow
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Step 1 & 2: Create generator and perform chunked counting
        counts_dict = count_points_in_polygons_in_chunks(
            load_preprocess_and_simplify_polygons(ROOT_FOLDER, TARGET_EXTENT, SIMPLIFY_TOLERANCE),
            CSV_PATH,
            LONGITUDE_COLUMN_NAME,
            LATITUDE_COLUMN_NAME,
            chunk_size=CSV_CHUNK_SIZE
        )

        # Step 3: Plot based on count results (needs to reload simplified polygons)
        plot_protected_areas_with_heat_from_counts(
            load_preprocess_and_simplify_polygons(ROOT_FOLDER, TARGET_EXTENT, SIMPLIFY_TOLERANCE),
            counts_dict
        )

    except ValueError as ve:
        print(f"\nError: {ve}")
    except FileNotFoundError as fnfe:
        print(f"\nFile Not Found Error: {fnfe}")
    except Exception as e:
        print(f"\nUnexpected Error: {str(e)}")