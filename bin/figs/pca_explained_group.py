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
    model_dir = '../core/model/model_90.0/color_reproduction_delay_unit/'
    sub_dir = '/noise_delta'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False
gen_data = True

prod_intervals=2000 # set the delay time to 800 ms for ploring the trajectory
pca_degree = np.arange(0, 360, 15) # Plot the trajectories of these colors
n_pca = 5
data_out_dir = './figs/fig_data/pca_explain.csv'

if gen_data:
    group = Agent_group(model_dir, rule_name, sub_dir = sub_dir)

    pca_label = []
    cum_explained_ratio = []

    mplot = MPloter()
    for sub in group.group:
        sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0, sigma_x=0)
        mplot.load_data(sub.fir_rate, sub.epochs, sub.behaviour['target_color'])
        mplot._pca_fit(n_pca, start_time=sub.epochs['stim1'][0] - 1, end_time=sub.epochs['response'][1])

        cum_explained_ratio_temp = np.cumsum(mplot.pca.explained_variance_ratio_)

        pca_label.extend(range(n_pca))
        cum_explained_ratio.extend(cum_explained_ratio_temp)


    pca_pd = pd.DataFrame({'pca_label': pca_label, 'cum_explained_ratio': cum_explained_ratio})

    pca_pd.to_csv(data_out_dir)

pca_pd = pd.read_csv(data_out_dir)

fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.2, 0.2, 0.65, 0.6])

label, mean_y, sd_y = tools.mean_se(pca_pd['pca_label'], pca_pd['cum_explained_ratio'], sd=True)
ax.scatter(label, mean_y, c='black')
# ax.plot(label, mean_y, c='black')
print('pca_explained:', mean_y)
print('sd error of pca_explained:', sd_y)
plt.errorbar(label, mean_y, yerr=sd_y, c='black') # the standard deviation is negnegiable (~10^-2).

ax.set_xticks([0, 2, 4])
ax.set_xticklabels([1, 3, 5])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Cummulative explained var. ratio')
ax.set_xlabel('PC')

fig.savefig('./figs/fig_collect/pca_explained.pdf', format='pdf')

plt.show()
