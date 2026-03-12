# Experiment 02: 6-Channel Crops CNN with Ordinal Regression

## Overview
Encodes each object type as a separate binary channel (6 total), preventing information mixing in the CNN's color space. Uses CORAL ordinal regression to exploit the ordered nature of difficulty ratings.


## Results
Val: Acc=0.2315 Off1=0.5694 MAE=1.55
Test: Acc=0.2656 Off1=0.5866 MAE=1.53

