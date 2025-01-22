# generate and save all manifold in directory
# show 3D manifold
import context
import errno
import numpy as np
import os

from core.data_plot import color_reproduction_dly_lib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from core.color_manager import Degree_color
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
from core.agent import Agent, Agent_group
import sys
import core.tools as tools

FONT_SIZE = 15

# Function to perform PCA analysis on RNN activity
def perform_pca_analysis(group, prod_intervals, pca_degree, n_pca):
    mplot = MPloter()  # Initialize manifold plotter.

    cum_explained_ratio_full = []  # To store cumulative explained variance ratios for full trials.
    cum_explained_ratio_delay = []  # To store cumulative explained variance ratios for delay epochs.
    pca_label_full = []
    pca_label_delay = []

    for sub in group.group:
        # Run experiments on the RNN with specified input colors
        sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0, sigma_x=0)
        mplot.load_data(sub.state, sub.epochs, sub.behaviour['target_color'])
        
        # PCA on full trial
        mplot._pca_fit(n_pca, start_time=sub.epochs['stim1'][0] - 1, end_time=sub.epochs['response'][1])
        pca_ratio_full_temp = np.cumsum(mplot.pca.explained_variance_ratio_)
        cum_explained_ratio_full.append(pca_ratio_full_temp)
        label_full = list(range(len(pca_ratio_full_temp)))
        pca_label_full.append(label_full)

        # PCA on delay epoch
        mplot._pca_fit(n_pca, start_time=sub.epochs['interval'][0] - 1, end_time=sub.epochs['interval'][1])
        pca_ratio_delay_temp = np.cumsum(mplot.pca.explained_variance_ratio_)
        cum_explained_ratio_delay.append(pca_ratio_delay_temp)
        label_delay = list(range(len(pca_ratio_delay_temp)))
        pca_label_delay.append(label_delay)

    # Combine PCA results across all RNNs
    pca_label_full = np.concatenate(pca_label_full)
    pca_label_delay = np.concatenate(pca_label_delay)
    cum_explained_ratio_full = np.concatenate(cum_explained_ratio_full)
    cum_explained_ratio_delay = np.concatenate(cum_explained_ratio_delay)

    return pca_label_full, cum_explained_ratio_full, pca_label_delay, cum_explained_ratio_delay

# Parameters for PCA analysis
rule_name = 'color_reproduction_delay_unit'
model_dir = '../core/model/model_90.0/color_reproduction_delay_unit/'  # Uniformed RNN
sub_dir = '/noise_delta'
gen_data = True
prod_intervals=1000   # Duration of the delay phase.
pca_degree = np.arange(0, 360, 100)  # Input colors for PCA.
n_pca = 5  # Number of principal components to analyze.

# Perform PCA analysis
group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
pca_label_full, cum_explained_ratio_full, pca_label_delay, cum_explained_ratio_delay = perform_pca_analysis(group, prod_intervals, pca_degree, n_pca)

# Calculate mean and standard deviation
label_full, mean_y_full, sd_y_full = tools.mean_se(pca_label_full, cum_explained_ratio_full, sd=True)
label_delay, mean_y_delay, sd_y_delay = tools.mean_se(pca_label_delay, cum_explained_ratio_delay, sd=True)

# Plot PCA results
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.2, 0.2, 0.65, 0.6])

# Plot full trial results
ax.errorbar(label_full, mean_y_full, yerr=sd_y_full, linestyle='-', c='tab:blue', label='Full Trial; Uniform')
ax.scatter(label_full, mean_y_full, c='tab:blue')

# Plot delay epoch results
ax.errorbar(label_delay, mean_y_delay, yerr=sd_y_delay, linestyle='dotted', c='tab:blue', label='Delay Epoch; Uniform')
ax.scatter(label_delay, mean_y_delay, c='tab:blue')

# Plot PCA results for biased RNNs
model_dir = '../core/model/model_12.5/color_reproduction_delay_unit/'
group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
pca_label_full, cum_explained_ratio_full, pca_label_delay, cum_explained_ratio_delay = perform_pca_analysis(group, prod_intervals, pca_degree, n_pca)

label_full, mean_y_full, sd_y_full = tools.mean_se(pca_label_full, cum_explained_ratio_full, sd=True)
label_delay, mean_y_delay, sd_y_delay = tools.mean_se(pca_label_delay, cum_explained_ratio_delay, sd=True)

ax.errorbar(label_full, mean_y_full, yerr=sd_y_full, linestyle='-', c='tab:red', label='Full Trial; Biased')
ax.scatter(label_full, mean_y_full, c='tab:red')
ax.errorbar(label_delay, mean_y_delay, yerr=sd_y_delay, linestyle='dotted', c='tab:red', label='Delay Epoch; Biased')
ax.scatter(label_delay, mean_y_delay, c='tab:red')

# global settings
ax.set_xticks([0, 2, 4])
ax.set_xticklabels([1, 3, 5])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('PC', fontsize=FONT_SIZE)
ax.set_ylabel('Cummulative explained var. ratio', fontsize=FONT_SIZE)
ax.tick_params(axis='x', labelsize=FONT_SIZE)
ax.tick_params(axis='y', labelsize=FONT_SIZE)

plt.legend()

fig.savefig('./figs/fig_collect/pca_explained.pdf', format='pdf')

plt.show()
