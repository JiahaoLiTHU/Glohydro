# Revealing Global Patterns of Hydropower Plants Using Remote Sensing

This repository contains the analysis scripts and model resources used in the study:

**_“Revealing Global Patterns of Hydropower Plants Using Remote Sensing”_**

Jiahao Li¹, Jiancheng Pan¹, Ramit Debnath², Dabo Guan¹, Xiaomeng Huang¹*, Qiang Zhang¹*

¹ Department of Earth System Science, Tsinghua University, Beijing, China
² Energy Policy Research Group, Judge Business School, University of Cambridge, UK

The project introduces **HydroVLM**, a vision–language framework for detecting hydropower plants (HPPs) globally using remote-sensing imagery, enabling systematic assessment of hydropower distribution, transboundary development, ecological conservation impacts, runoff alteration, and spatiotemporal dynamics.

---

## 🌍 Key Contributions

- 🛰 **HydroVLM model** for global top-down hydropower detection  
- 🌐 Identified **12,640 hydropower plants**, with **55.7% previously unreported**  
- 🗺 Comprehensive global analyses, including:  
  - Hydropower clustering and transboundary impacts  
  - Biodiversity & protected-area sensitivity  
  - Runoff alteration and flood-risk changes  
  - Spatiotemporal evolution of hydropower development  
  - Long-term biomass energy analysis  
- 📦 Modular repository for reproducible scientific workflows  

---

## 📁 Repository Structure

    .
    ├── Biomass anlysis
    │   └── analyse_historical_biomass.py
    │
    ├── Hydropower clusters
    │   └── Transboundary analysis.py
    │
    ├── Protected areas
    │   ├── bio_analysis_1.py
    │   ├── bio_analysis.py
    │   ├── protcet_area_hydro.py
    │   └── protect_area.py
    │
    ├── Runoff alteration and flood risk
    │   ├── flood_analysis.py
    │   ├── runoff_painting.py
    │   ├── streamflow_anly.py
    │   └── trend_analysis.py
    │
    ├── Spatiotemporal analysis
    │   ├── Analysis of longitude and latitude.py
    │   ├── country_analysis.py
    │   └── river.py
    │
    ├── Vlm
    │   └── ms-swift-main
    │
    └── README.md

---

## 📦 Module Descriptions

### 1. Biomass Analysis (`Biomass anlysis/`)

Analysis of historical biomass energy development.

- `analyse_historical_biomass.py` — Long-term biomass trend analysis and comparison with hydropower development.

---

### 2. Hydropower Clusters & Transboundary Impacts (`Hydropower clusters/`)

Hydropower clustering and transboundary river-basin assessment.

- `Transboundary analysis.py` — Identifies hydropower clusters in transboundary basins and evaluates upstream–downstream dependencies and impacts.

---

### 3. Protected Areas & Biodiversity (`Protected areas/`)

Assessment of hydropower interactions with ecological conservation zones and biodiversity.

- `bio_analysis.py` / `bio_analysis_1.py` — Biodiversity and ecological hotspot analysis related to hydropower plants  
- `protcet_area_hydro.py` / `protect_area.py` — Hydropower–protected-area interaction modeling and impact assessment  

---

### 4. Runoff Alteration & Flood Risk (`Runoff alteration and flood risk/`)

Hydrological impact evaluation of hydropower development.

- `flood_analysis.py` — Changes in downstream flood peaks and flood frequency  
- `runoff_painting.py` — Visualization of runoff alteration patterns  
- `streamflow_anly.py` — Streamflow pattern and regime analysis  
- `trend_analysis.py` — Long-term hydrological trend detection and attribution  

---

### 5. Spatiotemporal Analysis (`Spatiotemporal analysis/`)

Quantifying hydropower development patterns over space and time.

- `Analysis of longitude and latitude.py` — Spatial distribution metrics of hydropower plants  
- `country_analysis.py` — National-level hydropower characteristics and statistics  
- `river.py` — River-network-based hydropower structure and distribution analysis  

---

### 6. HydroVLM Vision–Language Model (`Vlm/ms-swift-main/`)

Core implementation of the vision–language model used for global hydropower detection from remote-sensing imagery.

- Model architecture and configuration  
- Feature extraction and inference utilities  
- Integration with hydropower detection workflows  

---

### Citation

If you are interested in the following work, please cite the following paper.

```

```