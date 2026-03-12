# Experiment 01: RGB Crops CNN

## Overview
Baseline CNN model predicting Geometry Dash level difficulty (1-10 stars) from RGB image crops. Each level is rendered as a color-coded image where object types are represented by distinct colors, then random fixed-size crops are sampled for training.

## Data Pipeline
```
levels/{n}stars/*.json          (raw level data)
    ↓  render_level()
data/plots/{n}stars/*.png       (RGB rendered images, scale=1)
    ↓  sample_crops()
data/crops/{n}stars/*.png       (128×64 crops, 5 per level)
data/crops/manifest.csv         (crop metadata)
```

## Color Encoding
| Object Group | RGB Color       | Visual     |
|-------------|-----------------|------------|
| Hazards     | (230, 50, 50)   | Red        |
| Blocks      | (160, 160, 160) | Gray       |
| Triggers    | (230, 210, 50)  | Yellow     |
| Portals     | (50, 220, 220)  | Cyan       |
| Orbs        | (50, 200, 80)   | Green      |
| Unknown     | (80, 80, 80)    | Dark Gray  |

## Model Architecture
```
Input: (3, 64, 128) RGB crop
Conv2d(3→32, 3×3) + ReLU + MaxPool2d(2)
Conv2d(32→64, 3×3) + ReLU + MaxPool2d(2)
Conv2d(64→128, 3×3) + ReLU + MaxPool2d(2)
AdaptiveAvgPool2d(4, 8)
Linear(4096→256) + ReLU + Dropout(0.3)
Linear(256→10)
Loss: CrossEntropyLoss
```

## How to Run
```bash
# Step 1: Prepare data
python prepare_data.py

# Step 2: Train
python train.py --epochs 30 --batch-size 32 --lr 1e-3
```

## Results
- **Test Accuracy (10-class):** ~21.5%
- Crop-level predictions are averaged (softmax mean) to produce level-level predictions.

## Key Findings
- Simple 3-layer CNN with RGB input serves as the project baseline.
- Information is lost because different object types share the RGB color space, causing CNN filters to mix semantic features during downsampling.
