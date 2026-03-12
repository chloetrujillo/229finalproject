# Experiment 01: RGB Crops CNN

## Overview
Baseline CNN model predicting Geometry Dash level difficulty (1-10 stars) from RGB image crops. Each level is rendered as a color-coded image where object types are represented by distinct colors, then random fixed-size crops are sampled for training.

## Results
Val: CropAcc=0.2639 LevelAcc=0.3102 LevelOff1=0.5648
Test: CropAcc=0.2739 LevelAcc=0.3048 LevelOff1=0.5727

