# Experiment 04: 6-Channel Weighted Crops CNN ⭐ (Best Model)

## Overview
Best-performing model. Renders each object type to its own binary channel with per-type stamp scaling — sparse but important types (hazards, portals, orbs) get 3× larger stamps so they survive CNN downsampling.

## Results
Val: Acc=0.2130 Off1=0.5648 MAE=1.68
Test: Acc=0.2240 Off1=0.5727 MAE=1.62

