# Introduction

This project explores unified dynamic model and computational principal for short-term memory. Codes were built upon on [Bi and Zhou, 2020](https://github.com/zedongbi/IntervalTiming).

# How to use this repo

Please find *./bin/PersiTransWM.ipynb* for reproducing main figures, and *./bin/PersiTransWM.ipynb* for SI figures. Pretrained models can be found [here](https://wustl.box.com/s/3xnt37fddxelvio2fztlawyieatf2agq)

## Requirement
    - Python version 3.8.8.
    - Pytorch 1.10.1; CUDA 11.3
    - colormath 3.0.0
    - mpi4py 3.1.3
    - pandas, seaborn, scikit-learn, statsmodels

# Comment

There are many hyperparameters in /core/default.py to control when the training stops, including stop\_cost, stop\_noise\_error, stop\_xxx etc. We set them to be very large so that they are not used in practice. As a result, the training always stops after achieving the min\_trials (2e5) when we observed relatively stable loss function and working memory error.

