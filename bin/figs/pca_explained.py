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

try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = '/noise_delta'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

prod_intervals=1000
pca_degree = np.arange(0, 360, 100) # Plot the trajectories of these colors
n_pca = 5

#sub = Agent(model_dir + sub_dir, rule_name=rule_name)
group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
mplot = MPloter()

pca_label_full = []
pca_label_delay = []
cum_explained_ratio_full = []
cum_explained_ratio_delay = []

for sub in group.group:
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0, sigma_x=0)
    mplot.load_data(sub.state, sub.epochs, sub.behaviour['target_color'])
    # full trial
    mplot._pca_fit(n_pca, start_time=sub.epochs['stim1'][0] - 1, end_time=sub.epochs['response'][1])
    pca_ratio_full_temp = np.cumsum(mplot.pca.explained_variance_ratio_)
    cum_explained_ratio_full.append(pca_ratio_full_temp)
    label_full = range(len(pca_ratio_full_temp))
    pca_label_full.append(label_full)

    # only delay
    mplot._pca_fit(n_pca, start_time=sub.epochs['interval'][0] - 1, end_time=sub.epochs['interval'][1])
    pca_ratio_delay_temp = np.cumsum(mplot.pca.explained_variance_ratio_)
    cum_explained_ratio_delay.append(pca_ratio_delay_temp)
    label_delay = range(len(pca_ratio_delay_temp))
    pca_label_delay.append(label_delay)

pca_label_full = np.array(pca_label_full).flatten()
pca_label_delay = np.array(pca_label_delay).flatten()
cum_explained_ratio_full = np.array(cum_explained_ratio_full).flatten()
cum_explained_ratio_delay = np.array(cum_explained_ratio_delay).flatten()

label_full, mean_y_full, sd_y_full = tools.mean_se(pca_label_full, cum_explained_ratio_full, sd=True)
label_delay, mean_y_delay, sd_y_delay = tools.mean_se(pca_label_delay, cum_explained_ratio_delay, sd=True)

fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.2, 0.2, 0.65, 0.6])

ax.errorbar(label_full, mean_y_full, yerr=sd_y_full)
print(sd_y_full)
ax.scatter(label_full, mean_y_full, label='Full Trial')
ax.errorbar(label_delay, mean_y_delay, yerr=sd_y_delay)
ax.scatter(label_delay, mean_y_delay, label='Delay Epoch')

ax.set_xticks([0, 2, 4])
ax.set_xticklabels([1, 3, 5])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Cummulative explained var. ratio')
ax.set_xlabel('PC')
plt.legend()

fig.savefig('./figs/fig_collect/pca_explained.pdf', format='pdf')

#plt.show()
