# Introduction

This project employed artificial recurrent neural networks (RNNs) as a model of the brain to explore the dynamic mechanism and computational principle of short-term memory.

# How to use this repo

Please find *./bin/PersiTransWM.ipynb* for reproducing main figures, and *./bin/PersiTransWM.ipynb* for SI figures.

Pretrained RNN models can be downloaded in [here](https://wustl.box.com/s/s2mm4h8pf0aurv04kfp4pwd75m5k98pn) The folder should be put under ./core/.

To draw figure without generating data (which take be hours), first, download figure data [here](https://wustl.box.com/s/s2mm4h8pf0aurv04kfp4pwd75m5k98pn) and put *fig_data* folder under ./bin/figs/. Next set *--gen_data* option in *./bin/PersiTransWM.ipynb* to false. Finally run the code.

## Requirement
    - Python version 3.8.8.
    - Pytorch 1.10.1; CUDA 11.3
    - colormath 3.0.0
    - mpi4py 3.1.3
    - pandas, seaborn, scikit-learn, statsmodels, numba
we also provided working_memory.yml for reference.

# Acknowledgement

The code is based on the code of [Bi and Zhou, 2020](https://github.com/zedongbi/IntervalTiming).

1. Bi, Z. & Zhou, C. Understanding the computation of time using neural network models. Proc. Natl. Acad. Sci. U. S. A. 117, 10530–10540 (2020).
