# Experiment 04: 6-Channel Weighted Crops CNN ⭐ (Best Model)

## Overview
Best-performing model. Renders each object type to its own binary channel with per-type stamp scaling — sparse but important types (hazards, portals, orbs) get 3× larger stamps so they survive CNN downsampling.

## Data Pipeline
```
levels/*.json
    ↓  render_weighted() — 6ch binary .npy
data/plots_weighted/{n}stars/*.npy   (variable size, 6 channels)
    ↓  sample_6ch_crops()
data/crops_weighted/{n}stars/*.npy   (64×128×6, 10 per level)
data/crops_weighted/manifest.csv
```

## Per-Type Stamp Scaling
| Channel | Object Group | Weight | Stamp Size (scale=2) |
|---------|-------------|--------|---------------------|
| 0       | Hazards     | 3×     | 6×6 pixels          |
| 1       | Blocks      | 1×     | 2×2 pixels          |
| 2       | Triggers    | 1×     | 2×2 pixels          |
| 3       | Portals     | 3×     | 6×6 pixels          |
| 4       | Orbs        | 3×     | 6×6 pixels          |
| 5       | Unknown     | 1×     | 2×2 pixels          |

## Model Architecture
```
Input: (6, 64, 386) merged weighted crops
Conv2d(6→16) + BN + ReLU + MaxPool2d(2)
Conv2d(16→32) + BN + ReLU + MaxPool2d(2)
Conv2d(32→64) + BN + ReLU + MaxPool2d(2)
AdaptiveAvgPool2d(4, 24)
Linear(6144→128) + ReLU
Dropout(0.3) + Linear(128→9) ordinal head
Loss: CORAL ordinal regression
Optimizer: Adam lr=1e-3, weight_decay=1e-4
Scheduler: ReduceLROnPlateau(factor=0.3, patience=3)
```

## How to Run
```bash
python prepare_data.py --scale 2 --n 10
python train.py --epochs 30 --batch-size 16 --lr 1e-3
```

## Results
- **Test Accuracy (10-class): 23.8%** ⭐
- **Off-by-1 Accuracy: 58.2%** ⭐
- **MAE: 1.61** ⭐

## Key Findings
- Stamp scaling is crucial: sparse types (hazards ~2.7% of objects) would vanish after 8× CNN downsampling without enlargement.
- This is the best model across all 10 experiments.
- Weighted rendering + ordinal regression + ReduceLROnPlateau all contribute to superior performance.
