# Experiment 03: Multi-Crop RGB CNN

## Overview
Instead of processing individual crops independently, this model merges 3 adjacent crops (sorted by x-coordinate) into a wider image using a sliding window, giving the CNN more spatial context. Uses CrossEntropy loss.

## Results
Val: Acc=0.2708 Off1=0.5023 MAE=2.01
Test: Acc=0.2725 Off1=0.4988 MAE=2.00

