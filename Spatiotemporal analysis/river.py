import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

warnings.filterwarnings('ignore')


def plot_rivers_only():
    """
    Plot an elliptical world map containing only major rivers
    """
    # --------------------------
    # Configuration Parameters
    # --------------------------
    RIVER_SHP_PATH = "./HydroRIVERS_v10_shp/HydroRIVERS_v10_shp/HydroRIVERS_v10.shp"
    OUTPUT_PATH = './rivers_only.png'

    # River color and style scheme (sorted by discharge from largest to smallest)
    RIVER_STYLE_MAP = [
        (100000, 3.5, '#00274C'),  # ≥100,000 m³/s: Extra Large Rivers
        (10000, 2.8, '#003F6B'),   # 10,000-99,999 m³/s: Large Rivers
        (1000, 2.0, '#005B96'),    # 1,000-9,999 m³/s: Medium Rivers
        (100, 1.5, '#0077B6'),     # 100-999 m³/s: Small-Medium Rivers
        (10, 1.0, '#0096C7'),      # 10-99 m³/s: Small Rivers
    ]

    # --------------------------
    # 1. Data Preparation
    # --------------------------
    print("Loading river data...")
    # Read river data with filtering
    rivers_gdf = gpd.read_file(
        RIVER_SHP_PATH,
        usecols=["ORD_STRA", "DIS_AV_CMS", "geometry"],
        where="ORD_STRA >= 5",  # Keep only medium and larger rivers (stream order ≥5)
        engine="pyogrio"
    )

    # Further filter rivers by discharge
    rivers_gdf = rivers_gdf[rivers_gdf["DIS_AV_CMS"] > 50]  # Discharge > 50 m³/s

    # Calculate centroid latitude to exclude high-latitude areas
    rivers_gdf['centroid'] = rivers_gdf.geometry.centroid
    rivers_gdf['latitude'] = rivers_gdf['centroid'].y
    rivers_gdf = rivers_gdf[rivers_gdf['latitude'] < 90]  # Exclude extreme high latitudes

    # Simplify geometry to reduce file size and improve rendering speed
    rivers_gdf["geometry"] = rivers_gdf["geometry"].simplify(tolerance=0.002, preserve_topology=True)

    print(f"Number of rivers after filtering: {len(rivers_gdf)}")

    # --------------------------
    # 2. Create Figure and Map
    # --------------------------
    plt.figure(figsize=(20, 12))

    # Use Robinson projection (elliptical world map)
    projection = ccrs.Robinson()
    ax = plt.axes(projection=projection)

    # Set global extent
    ax.set_global()

    # --------------------------
    # 3. Configure Map Style
    # --------------------------
    # Map background color
    ax.set_facecolor('#f9f9f9')
    # Land color - light gray
    ax.add_feature(cfeature.LAND, facecolor='#e5e5e5', zorder=1)
    # Ocean color - white
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=1)

    # Set map extent (exclude Antarctica)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    # --------------------------
    # 4. Plot Rivers (on Elliptical Projection)
    # --------------------------
    print("Plotting river network...")

    def get_flow_category(flow_value):
        """Determine river category based on average discharge"""
        for i, (min_flow, _, _) in enumerate(RIVER_STYLE_MAP):
            if flow_value >= min_flow:
                return i
        return len(RIVER_STYLE_MAP) - 1  # Default to smallest category if below all thresholds

    # Add flow category column to GeoDataFrame
    rivers_gdf['flow_category'] = rivers_gdf['DIS_AV_CMS'].apply(get_flow_category)

    # Plot rivers from largest to smallest discharge category
    for cat_idx, (min_flow, linewidth, color) in enumerate(RIVER_STYLE_MAP):
        cat_rivers = rivers_gdf[rivers_gdf['flow_category'] == cat_idx]

        if len(cat_rivers) > 0:
            print(f"Plotting rivers with discharge ≥{min_flow} m³/s: {len(cat_rivers)} rivers")

            # Plot each river on the elliptical projection
            for idx, river in cat_rivers.iterrows():
                # Transform geometry to target projection (Robinson)
                transformed_geom = projection.project_geometry(
                    river.geometry,
                    ccrs.PlateCarree()  # Source CRS: WGS84 (default for HydroRIVERS)
                )

                if transformed_geom is not None and not transformed_geom.is_empty:
                    # Plot individual river
                    ax.add_geometries(
                        [transformed_geom],
                        crs=projection,
                        facecolor='none',
                        edgecolor=color,
                        linewidth=linewidth,
                        alpha=0.8,
                        zorder=2  # Rivers appear above land
                    )

    # Turn off axis labels and ticks
    ax.axis('off')

    # --------------------------
    # 5. Save Figure
    # --------------------------
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close figure to free memory

    print(f"River map saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_rivers_only()