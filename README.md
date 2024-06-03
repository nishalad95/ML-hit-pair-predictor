### ML Hit Pair Predictor

A ML-based algorithm for predicting whether doublet hit pairs belong to the same track, with the aim of reducing the number of fake triplet seeds in the HLT. The algorithm utilizes a Bayes classifier with Kernel Density Estimate, and is trained on ATLAS MC data.

#### Training Pixel-Barrel Doublets:

`python train_validate_barrel.py -d <training data filepath> -b barrel_optimum_kde_bandwidths.csv`
d: training data file format in `tar.gz` contaning csv files from MC simulation
t: (bool) triplet validation stage, default value: `False`

#### Training Pixel-Endcap Doublets:

`python train_validate_endcap.py -d <training data filepath> -b endcap_optimum_kde_bandwidths.csv`
d: training data file format in `tar.gz` contaning csv files from MC simulation
t: (bool) triplet validation stage, default value: `False`


