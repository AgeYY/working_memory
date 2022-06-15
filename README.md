# Introduction

This project explores unified dynamic model and computational principal for short-term memory. 
This project emplo

# How to use this repo

Please find *./bin/PersiTransWM.ipynb* for reproducing main figures, and *./bin/PersiTransWM.ipynb* for SI figures. Pretrained models can be downloaded in [here](https://wustl.box.com/s/3xnt37fddxelvio2fztlawyieatf2agq) which should be put under ./core/

## Requirement
    - Python version 3.8.8.
    - Pytorch 1.10.1; CUDA 11.3
    - colormath 3.0.0
    - mpi4py 3.1.3
    - pandas, seaborn, scikit-learn, statsmodels, numba

# Comment

There are many stop training conditions in /core/default.py, including stop\_cost, stop\_noise\_error, stop\_xxx etc. We set them properly so that most of them are not used in practice. As a result, the training always terminates after achieving the min\_trials (2e5) trials when we observed a relatively stable working memory error. The purpose of this project is not persuing high memory accuracy anyway.

bump.py

# Acknowledgement

We thanks [Bi and Zhou, 2020](https://github.com/zedongbi/IntervalTiming) where our codes were built upon on.

1. Bi, Z. & Zhou, C. Understanding the computation of time using neural network models. Proc. Natl. Acad. Sci. U. S. A. 117, 10530–10540 (2020).
