# Introduction

This project employed artificial recurrent neural networks (RNNs) as a model of the brain to explore the dynamic mechanism and computational principle of short-term memory.

# How to use this repo

Please find *./bin/PersiTransWM.ipynb* for reproducing main figures, and *./bin/PersiTransWM.ipynb* for SI figures. Pretrained models can be downloaded in [here](https://wustl.box.com/s/3xnt37fddxelvio2fztlawyieatf2agq) which should be put under ./core/

## Requirement
    - Python version 3.8.8.
    - Pytorch 1.10.1; CUDA 11.3
    - colormath 3.0.0
    - mpi4py 3.1.3
    - pandas, seaborn, scikit-learn, statsmodels, numba

# Acknowledgement

The code is based on the code of [Bi and Zhou, 2020](https://github.com/zedongbi/IntervalTiming).

1. Bi, Z. & Zhou, C. Understanding the computation of time using neural network models. Proc. Natl. Acad. Sci. U. S. A. 117, 10530–10540 (2020).
