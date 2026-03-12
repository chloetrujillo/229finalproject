# Experiment 06: RGB Full-Level CNN

## Overview
Resizes the entire level to a fixed 64x512 RGB image, preserving global spatial structure. Uses semantic color encoding (same as exp01) but with full-level view instead of crops.

## Results
- **Test Accuracy:** ~22% | **Off-by-1:** ~53% | **MAE:** ~1.85
(with lr = 1e-3, I didn't rerun this one, but the others are with 1e-4)
