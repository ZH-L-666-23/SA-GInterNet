# SA-GInterNet
# SA-GInterNet
# ST-CGSAN: Spatio-Temporal Cross-Graph Spatial Attention Network

## Overview

ST-CGSAN is a novel spatio-temporal graph neural network designed for atmospheric CO₂ concentration prediction. The model integrates two key branches:

1. **Structure-Aware Spatial Modeling Branch**
2. **Cross-Graph Interaction Temporal Modeling Branch**

This repository contains the complete implementation of the ST-CGSAN model along with baseline experiments, ablation studies, and variable importance analysis.

## Project Structure

```
STCGSANproject/
├── data/                    # Data directory
│   ├── raw/                 # Raw data (satellite/covariates)
│   ├── processed/           # Preprocessed CSV data (48 files, 2017-2020)
│   └── Graph/               # Graph construction outputs (48 spatial graphs)
├── src/                     # Source code
│   ├── models/              # ST-CGSAN model implementation
│   ├── baselines/           # Baseline models
│   ├── experiments/         # Experiment runners
│   ├── graph_construction/  # Graph construction utilities
│   ├── data_preprocessing/  # Data preprocessing scripts
│   └── utils/               # Utility functions
├── results/                 # Experiment results
│   ├── main/                # Main model results
│   ├── baselines/           # Baseline comparison results
│   ├── ablation/            # Ablation study results
│   └── variable_importance/ # Variable importance analysis
└── run_*.py                 # Execution scripts
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- numpy, pandas, scikit-learn, scipy
- xgboost (for baseline experiments)

## Data Description

### Preprocessed Data (`data/processed/`)

48 CSV files covering 2017-2020 monthly data for Hunan, Hubei, Anhui, Jiangxi provinces:
- File naming: `CO2_Hunan_Hubei_Anhui_Jiangxi_YYYY-MM.csv`
- Columns: `sample_id, lon, lat, date, province, XCO2, NTL, TEM, SP, PRE, ET, UW, VW, DSR, AOD, NDVI, CO, NO2, SO2, O3, DEM`

### Graph Data (`data/Graph/`)

48 spatial graphs and cross-graph sequences:
- `graph_YYYY-MM.npz`: Monthly spatial graphs
- `cross_graph_sequences.npz`: Temporal cross-graph sequences

## Model Architecture

### ST-CGSAN (`src/models/st_cgsan.py`)

#### Structure-Aware Spatial Branch (`src/models/structure_aware_spatial.py`)
- Multi-head graph attention mechanism
- Captures spatial dependencies between nodes
- Structure-aware representation learning

#### Cross-Graph Temporal Branch (`src/models/cross_graph_temporal.py`)
- Cross-graph interaction mechanism
- Models temporal dependencies across consecutive graphs
- Maintains temporal consistency

#### Fusion Layer
- Concatenation-based fusion of spatial and temporal features
- Attention-based fusion option available

## Experiments

### 1. Baseline Comparison (`run_baselines.py`)

8 baseline models implemented:
| Model | RMSE | MAE |
|-------|------|-----|
| RF | 2.85 | 2.21 |
| XGBoost | 2.67 | 2.08 |
| CNN | 2.31 | 1.81 |
| LSTM | 2.15 | 1.69 |
| CNN-LSTM | 2.09 | 1.64 |
| GCN | 2.24 | 1.75 |
| GAT | 2.18 | 1.71 |
| STGCN | 2.01 | 1.58 |


### 2. Ablation Study (`run_ablation.py`)

5 ablation configurations:
| Configuration | Description | RMSE | MAE |
|---------------|-------------|------|-----|
| full_model | Complete ST-CGSAN | 1.82 | 1.45 |
| no_spatial_branch | Remove spatial branch | 2.10 | 1.65 |
| no_temporal_branch | Remove temporal branch | 2.03 | 1.59 |
| no_structure_aware | Remove structure-aware module | 1.95 | 1.51 |
| no_cross_graph_interaction | Remove cross-graph interaction | 2.01 | 1.57 |

### 3. Variable Importance Analysis (`run_variable_importance.py`)

 importance methods:
- Permutation Importance

Top 5 important features:
1. NTL (Nighttime Lights)
2. TEM (Temperature)
3. AOD (Aerosol Optical Depth)
4. NDVI (Normalized Difference Vegetation Index)
5. DSR (Downward Shortwave Radiation)

## Running Instructions

### Data Preprocessing
```bash
python data/processed/generate_all_csv_files.py
```

### Graph Construction
```bash
python data/Graph/build_graphs.py
```

### Baseline Experiments
```bash
python run_baselines.py
```

### Ablation Study
```bash
python run_ablation.py
```

### Variable Importance Analysis
```bash
python run_variable_importance.py
```

## Result Files

### Baseline Results (`results/baselines/`)
- Per-model prediction CSV files
- Per-model metrics CSV files
- `baseline_summary.csv`: Overall comparison

### Ablation Results (`results/ablation/`)
- Per-configuration prediction CSV files
- Per-configuration metrics CSV files
- `ablation_summary.csv`: Ablation comparison

### Variable Importance Results (`results/variable_importance/`)
- `permutation_importance.csv`



## License

This project is for research purposes only.
