# Experiment 03: Multi-Crop RGB CNN

## Overview
Instead of processing individual crops independently, this model merges 3 adjacent crops (sorted by x-coordinate) into a wider image using a sliding window, giving the CNN more spatial context. Uses CrossEntropy loss.

## Data Pipeline
```
levels/*.json → data/crops/*.png  (same as exp01)
    ↓  sliding window merge (3 crops → 386px wide)
Input: (3, 64, 386) merged RGB crops per level
    ↓  CNN backbone per merged crop
    ↓  mask-aware mean pooling across all merged views
Output: 10-class prediction
```

## Model Architecture
```
Input: (B, M, 3, 64, 386)  — M merged crops per level
Conv2d(3→32) + BN + ReLU + MaxPool2d(2)
Conv2d(32→64) + BN + ReLU + MaxPool2d(2)
Conv2d(64→128) + BN + ReLU + MaxPool2d(2)
AdaptiveAvgPool2d(4, 8)
Linear(4096→256) + ReLU → per-crop features
Mean pooling with mask → level feature
Dropout(0.3) + Linear(256→10)
Loss: CrossEntropyLoss
```

## How to Run
```bash
python prepare_data.py   # Same data as exp01
python train.py --epochs 30 --batch-size 16 --lr 1e-3
```

## Results
- **Test Accuracy:** ~21%

## Key Findings
- Sliding window merging provides more context than single crops but marginal improvement over exp01.
- Mean pooling across variable-length merged crops allows batch processing.
