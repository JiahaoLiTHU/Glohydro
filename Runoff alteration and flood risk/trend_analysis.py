# -*- coding: utf-8 -*-
"""
Plot Distribution of Tau Values for Different Trend Types Using Real Data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---

# 1. Data file path (Please verify the path is correct)
DATA_FILE_PATH = './trend_with_coordinates_full.csv'

# 2. Color scheme
COLORS = {
    'Significant Increasing': '#80a6e2',  # Significant increasing trend (blue)
    'No Significant Trend': '#F8E9D6',    # No significant trend (beige)
    'Significant Decreasing': '#cf3d3e'   # Significant decreasing trend (red)
}

# 3. Chart output path and settings
OUTPUT_DIR = './visualizations'
OUTPUT_FILE_NAME = 'tau_value_distribution_english.png'
FIGURE_SIZE = (12, 4)
DPI = 300


def main():
    """Main function"""
    print("--- Starting to plot Tau Value Distribution (English Version) ---")

    # 1. Check and create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 2. Load data
    try:
        print(f"Reading data file: {DATA_FILE_PATH}")
        df = pd.read_csv(DATA_FILE_PATH)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found. Please check if the path '{DATA_FILE_PATH}' is correct.")
        return
    except Exception as e:
        print(f"Error occurred while reading the file: {e}")
        return

    # 3. Data preprocessing and validation
    required_columns = ['Trend', 'Tau']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Data file is missing required columns. Need '{', '.join(required_columns)}', but only found '{', '.join(df.columns)}'.")
        return

    # Ensure Tau column is numeric
    df['Tau'] = pd.to_numeric(df['Tau'], errors='coerce')
    # Remove rows with missing Tau values or Trend labels
    df.dropna(subset=['Tau', 'Trend'], inplace=True)

    print("\nData Basic Information:")
    print(f"- Total number of records: {len(df)}")
    print("- Trend type distribution:")
    trend_counts = df['Trend'].value_counts()
    print(trend_counts)

    # 4. Configure matplotlib parameters
    plt.rcParams['font.family'] = 'Arial'  # Use Arial font
    plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

    # 5. Create figure
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = plt.gca()

    # 6. Plot histogram
    print("\nPlotting histogram...")
    # Automatically calculate appropriate bin width
    bin_width = 0.02
    bins = np.arange(df['Tau'].min() - bin_width, df['Tau'].max() + bin_width, bin_width)

    # Iterate through trend types in color scheme
    for trend_type, color in COLORS.items():
        if trend_type in df['Trend'].values:
            tau_data = df[df['Trend'] == trend_type]['Tau']
            # Plot histogram (no label since legend is disabled)
            ax.hist(tau_data, bins=bins, alpha=0.8, color=color,
                    edgecolor='white', linewidth=0.8)
            print(f"- Plotted distribution for '{trend_type}'")
        else:
            print(f"- Warning: '{trend_type}' not found in data, skipped.")

    # 7. Add chart elements
    # Vertical line at x=0 (no legend label)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.8)
    # Uncomment below if title is needed
    # ax.set_title('Distribution of Tau Values for Different Runoff Trends',
    #              fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Tau Value',
                  fontsize=18, fontweight='bold')
    ax.set_ylabel('Frequency',
                  fontsize=18, fontweight='bold')

    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Optimize x-axis ticks and range
    ax.set_xlim(left=df['Tau'].min() - 0.05, right=df['Tau'].max() + 0.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=False, prune='both', nbins=15))

    # Set tick label font to 16pt bold
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # --- Added Code ---
    # Enhance frame style to match histogram edge width
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # --- End of Added Code ---

    plt.tight_layout()

    # 8. Save chart
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"\nChart successfully saved to: {output_path}")
    print("--- Task Completed ---")


if __name__ == "__main__":
    main()