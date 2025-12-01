"""
Global Hydropower Plants Latitude & Longitude Distribution Boxplots
Academic Version: Y-axis Label Adjusted, 26pt Bold Font, 45° X-ticks
Local Path: ./hydro_final_new.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# Core Configuration (Y-label Adjusted)
# ----------------------
# Local file paths (matches your system)
DATA_PATH = ''
LATITUDE_OUTPUT = ''
LONGITUDE_OUTPUT = ''

# Figure style settings (16:9, academic standard)
FIG_SIZE = (12, 9)  # Maintained aspect ratio
FONT_SIZE = 26       # Kept your 26pt font setting
FONT_WEIGHT = 'bold'
DPI = 300            # High definition for papers
X_TICK_ROTATION = 45 # 45° X-ticks for full name display
Y_LABEL_PAD = 15     # Reduced padding (key adjustment: makes Y-label closer to axis)

# Soft vibrant Morandi colors (academic-friendly)
SOFT_VIBRANT_MORANDI = {
    'Asia': '#FF9E9E',       # Soft coral
    'Europe': '#82E0AA',     # Soft teal-green
    'North America': '#FFE066', # Soft yellow
    'South America': '#78E08F', # Soft mint green
    'Africa': '#74B9FF',     # Soft sky blue
    'Oceania': '#A29BFE'     # Soft lavender
}

def assign_continent(lat, lon):
    """Assign continent based on latitude and longitude coordinates"""
    if (-170 <= lon <= -50) and (10 <= lat <= 85):
        return 'North America'
    elif (-80 <= lon <= -30) and (-56 <= lat <= 15):
        return 'South America'
    elif (-10 <= lon <= 40) and (35 <= lat <= 70):
        return 'Europe'
    elif (25 <= lon <= 180) and (-10 <= lat <= 75):
        return 'Asia'
    elif (-17 <= lon <= 52) and (-38 <= lat <= 37):
        return 'Africa'
    elif (110 <= lon <= 180) and (-50 <= lat <= -10):
        return 'Oceania'
    else:
        return 'Others'

def create_latitude_boxplot(df):
    """Create latitude boxplot (Y-label closer to axis)"""
    # Prepare data
    continents_order = ['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Oceania']
    lat_data = []
    valid_continents = []

    for continent in continents_order:
        continent_data = df[df['Continent'] == continent]
        if len(continent_data) > 0:
            lat_data.append(continent_data['Latitude'].values)
            valid_continents.append(continent)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)

    # Generate boxplot
    bp = ax.boxplot(
        lat_data,
        tick_labels=valid_continents,
        patch_artist=True,
        boxprops=dict(linewidth=2.5),
        whiskerprops=dict(linewidth=2.5),
        capprops=dict(linewidth=2.5),
        medianprops=dict(linewidth=3.5, color='#2D2D2D'),
        flierprops=dict(
            marker='o',
            markerfacecolor='white',
            markersize=5,
            markeredgecolor='#555555',
            markeredgewidth=1.5
        )
    )

    # Apply colors
    for patch, continent in zip(bp['boxes'], valid_continents):
        patch.set_facecolor(SOFT_VIBRANT_MORANDI[continent])
        patch.set_alpha(0.8)

    # Set Y-axis label (KEY CHANGE: Y_LABEL_PAD=15, 60→15)
    ax.set_ylabel(
        'Latitude (°)',
        fontsize=FONT_SIZE,
        fontweight=FONT_WEIGHT,
        labelpad=Y_LABEL_PAD  # Reduced padding to bring label closer to Y-axis
    )
    ax.set_xlabel('')  # Keep no X-axis title

    # Set tick styles
    ax.tick_params(
        axis='x',
        labelsize=FONT_SIZE,
        width=2.5,
        length=10,
        pad=26  # Keep X-tick padding for rotated labels
    )
    ax.tick_params(
        axis='y',
        labelsize=FONT_SIZE,
        width=2.5,
        length=10,
        pad=15  # Reduced Y-tick padding (optional: aligns tick labels with axis)
    )

    # Set tick label styles
    for label in ax.get_xticklabels():
        label.set_fontweight(FONT_WEIGHT)
        label.set_fontsize(FONT_SIZE)
        label.set_rotation(X_TICK_ROTATION)
        label.set_ha('right')

    for label in ax.get_yticklabels():
        label.set_fontweight(FONT_WEIGHT)
        label.set_fontsize(FONT_SIZE)

    # Axis limits and grid
    ax.set_ylim(-70, 90)
    ax.grid(
        True,
        alpha=0.4,
        linestyle='--',
        linewidth=2,
        color='#CCCCCC'
    )

    # Clean layout
    ax.set_title('')
    if ax.legend_:
        ax.legend_.remove()

    # Save plot
    plt.tight_layout()
    plt.savefig(
        LATITUDE_OUTPUT,
        dpi=DPI,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()
    print(f"✅ Latitude plot saved: {LATITUDE_OUTPUT}")

def create_longitude_boxplot(df):
    """Create longitude boxplot (same Y-label adjustment)"""
    # Prepare data
    continents_order = ['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Oceania']
    lon_data = []
    valid_continents = []

    for continent in continents_order:
        continent_data = df[df['Continent'] == continent]
        if len(continent_data) > 0:
            lon_data.append(continent_data['Longitude'].values)
            valid_continents.append(continent)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)

    # Generate boxplot
    bp = ax.boxplot(
        lon_data,
        tick_labels=valid_continents,
        patch_artist=True,
        boxprops=dict(linewidth=2.5),
        whiskerprops=dict(linewidth=2.5),
        capprops=dict(linewidth=2.5),
        medianprops=dict(linewidth=3.5, color='#2D2D2D'),
        flierprops=dict(
            marker='o',
            markerfacecolor='white',
            markersize=5,
            markeredgecolor='#555555',
            markeredgewidth=1.5
        )
    )

    # Apply colors
    for patch, continent in zip(bp['boxes'], valid_continents):
        patch.set_facecolor(SOFT_VIBRANT_MORANDI[continent])
        patch.set_alpha(0.8)

    # Set Y-axis label (Same adjustment: labelpad=15)
    ax.set_ylabel(
        'Longitude (°)',
        fontsize=FONT_SIZE,
        fontweight=FONT_WEIGHT,
        labelpad=Y_LABEL_PAD  # Reduced padding for closer alignment
    )
    ax.set_xlabel('')  # Keep no X-axis title

    # Set tick styles
    ax.tick_params(
        axis='x',
        labelsize=FONT_SIZE,
        width=2.5,
        length=10,
        pad=26
    )
    ax.tick_params(
        axis='y',
        labelsize=FONT_SIZE,
        width=2.5,
        length=10,
        pad=15  # Reduced Y-tick padding
    )

    # Set tick label styles
    for label in ax.get_xticklabels():
        label.set_fontweight(FONT_WEIGHT)
        label.set_fontsize(FONT_SIZE)
        label.set_rotation(X_TICK_ROTATION)
        label.set_ha('right')

    for label in ax.get_yticklabels():
        label.set_fontweight(FONT_WEIGHT)
        label.set_fontsize(FONT_SIZE)

    # Axis limits and grid
    ax.set_ylim(-200, 200)
    ax.grid(
        True,
        alpha=0.4,
        linestyle='--',
        linewidth=2,
        color='#CCCCCC'
    )

    # Clean layout
    ax.set_title('')
    if ax.legend_:
        ax.legend_.remove()

    # Save plot
    plt.tight_layout()
    plt.savefig(
        LONGITUDE_OUTPUT,
        dpi=DPI,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()
    print(f"✅ Longitude plot saved: {LONGITUDE_OUTPUT}")

def main():
    """Main execution workflow"""
    try:
        # Load data
        print("🔍 Loading hydropower data...")
        df = pd.read_csv(DATA_PATH)

        # Validate columns
        required_cols = ['Latitude', 'Longitude']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        print(f"✅ Loaded {len(df)} hydropower plant records")

        # Assign continents
        print("🌍 Assigning continents based on coordinates...")
        df['Continent'] = df.apply(
            lambda row: assign_continent(row['Latitude'], row['Longitude']),
            axis=1
        )

        # Filter continents
        main_continents = ['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Oceania']
        df_filtered = df[df['Continent'].isin(main_continents)]
        print(f"✅ Filtered to {len(df_filtered)} records across 6 continents")

        # Generate plots
        print("\n🎨 Creating latitude boxplot (Y-label adjusted)...")
        create_latitude_boxplot(df_filtered)

        print("🎨 Creating longitude boxplot (Y-label adjusted)...")
        create_longitude_boxplot(df_filtered)

        # Success message
        print("\n🎉 All plots generated successfully (Y-label closer to axis)!")
        print(f"📁 Latitude plot: {LATITUDE_OUTPUT}")
        print(f"📁 Longitude plot: {LONGITUDE_OUTPUT}")

    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        print("Please confirm the file path is correct.")
    except Exception as e:
        print(f"❌ Execution error: {str(e)}")

if __name__ == "__main__":
    # Uncomment to install dependencies (run once if missing)
    # import subprocess
    # import sys
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "matplotlib"])

    main()