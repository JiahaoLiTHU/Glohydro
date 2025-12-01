#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🌿 全部水电站年度生物量趋势分析脚本（最终版）
自动遍历目录下所有水电站（按文件名前缀区分）
功能：
  - 自动识别每个水电站的多年份文件
  - 分别计算各水电站年度生物量
  - 生成单站趋势图 + 年度分布图 + 汇总CSV
作者: ChatGPT (2025)
"""

# ==========================
# 1️⃣ 导入依赖
# ==========================
import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from collections import defaultdict


# ==========================
# 2️⃣ 读取第65波段
# ==========================
def load_biomass_band(path, band_index=65):
    print(f"📥 Loading GeoTIFF: {path}")
    with rasterio.open(path) as src:
        bands = src.count
        if band_index > bands:
            raise ValueError(f"❌ 文件只有 {bands} 个波段，无法读取第 {band_index} 波段。")
        biomass = src.read(band_index).astype(np.float32)
        profile = src.profile
    print(f"  ✅ 波段数: {bands}, 形状: {biomass.shape}, mean={np.nanmean(biomass):.2f}")
    return biomass, profile


# ==========================
# 3️⃣ 计算总生物量
# ==========================
def compute_total_biomass(biomass, profile, fallback_pixel_size_m=100):
    transform = profile.get("transform", None)
    valid = ~np.isnan(biomass)

    if transform is None or getattr(transform, "a", 0) == 0 or getattr(transform, "e", 0) == 0:
        print("  ⚠️ 无有效地理变换，使用相对像元求和（非绝对量）")
        return float(np.nansum(biomass[valid]))

    pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    if pixel_area_m2 < 1e-6:
        print("  ⚠️ 像元面积过小或无效，使用 fallback 像元大小计算")
        pixel_area_m2 = fallback_pixel_size_m ** 2

    pixel_area_ha = pixel_area_m2 / 10000.0
    total_mg = float(np.nansum(biomass[valid]) * pixel_area_ha)
    return total_mg


# ==========================
# 4️⃣ 可视化函数
# ==========================
def visualize_biomass(biomass, station_id, year, out_dir, vmin=0, vmax=300):
    biomass = np.clip(biomass, vmin, vmax)
    mean_val, med_val = np.nanmean(biomass), np.nanmedian(biomass)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(biomass, cmap="YlGn", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Biomass (Mg/ha)")
    plt.title(f"{station_id} - Biomass {year}")
    plt.axis("off")
    plt.text(0.02, 0.98, f"Mean: {mean_val:.2f}\nMedian: {med_val:.2f}",
             transform=plt.gca().transAxes, va="top", fontsize=9,
             bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{station_id}_biomass_{year}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_total_trend(station_id, years, totals, out_dir):
    plt.figure(figsize=(7,5))
    plt.plot(years, totals, marker="o", linewidth=2, color="#2E8B57")
    plt.xlabel("Year")
    plt.ylabel("Total Biomass (Mg)")
    plt.title(f"Total Biomass Trend - {station_id}")
    plt.grid(True, linestyle=":", alpha=0.6)
    for i, v in enumerate(totals):
        plt.text(years[i], v, f"{v/1e6:.2f}M", ha="center", va="bottom", fontsize=8)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{station_id}_trend.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ==========================
# 5️⃣ 主分析函数
# ==========================
def analyze_all_stations(data_dir, out_dir, band_index=65):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # 按水电站分组文件
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".tif")]
    pattern = re.compile(r"(GHR\d+)_Stacked_(\d{4})\.tif")

    station_files = defaultdict(list)
    for f in all_files:
        m = pattern.match(f)
        if m:
            station_id, year = m.group(1), int(m.group(2))
            station_files[station_id].append((year, os.path.join(data_dir, f)))

    print(f"🔍 检测到 {len(station_files)} 个水电站。")

    summary = []

    # 循环每个水电站
    for station_id, year_paths in sorted(station_files.items()):
        print(f"\n🏞️ 分析水电站: {station_id}")
        year_paths.sort(key=lambda x: x[0])
        totals = []
        years = []

        for year, path in year_paths:
            biomass, profile = load_biomass_band(path, band_index)
            total = compute_total_biomass(biomass, profile)
            totals.append(total)
            years.append(year)
            print(f"  ✅ {year}: {total:,.2f} Mg")
            visualize_biomass(biomass, station_id, year,
                              os.path.join(fig_dir, station_id))

            summary.append({"station": station_id, "year": year, "total_biomass_Mg": total})

        # 趋势图
        plot_total_trend(station_id, years, totals, os.path.join(fig_dir, station_id))

    # 汇总 CSV
    csv_path = os.path.join(out_dir, "all_stations_biomass_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["station", "year", "total_biomass_Mg"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(f"\n📄 全部水电站年度生物量汇总已保存: {csv_path}")
    print("🌿 全部水电站年度趋势分析完成。")


# ==========================
# 6️⃣ 命令行入口
# ==========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="分析所有水电站的年度生物量变化趋势")
    ap.add_argument("--data_dir", default="./datasets/glohydro_dataset/embdings_abdg", help="存放年度 65 波段 GeoTIFF 的文件夹")
    ap.add_argument("--out_dir", default="./outputs_all_stations", help="输出结果文件夹")
    ap.add_argument("--band_index", type=int, default=65, help="生物量所在波段索引（默认65）")
    args = ap.parse_args()

    analyze_all_stations(args.data_dir, args.out_dir, args.band_index)
