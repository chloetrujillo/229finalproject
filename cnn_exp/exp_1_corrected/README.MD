# Experiment 01 corrected: RGB Crops CNN with MSE loss

## Overview
Baseline CNN model predicting Geometry Dash level difficulty (1-10 stars) from RGB image crops. Each level is rendered as a color-coded image where object types are represented by distinct colors, then random fixed-size crops are sampled for training.

## Results

Val: CropAcc=0.1667 LevelAcc=0.1551 LevelOff1=0.5023
Test: CropAcc=0.1607 LevelAcc=0.1709 LevelOff1=0.4896