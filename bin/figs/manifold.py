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
from core.agent import Agent
import sys
import core.tools as tools


# Try to read command-line arguments or set default values
try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model/model_12.5/color_reproduction_delay_unit/'
    sub_dir = '/model_5/noise_delta'

# Determine whether to generate new data
try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False
gen_data = True

# Configuration for trajectory visualization
prod_intervals=800 # set the delay time to 800 ms for ploring the trajectory
pca_degree = np.arange(0, 360, 20) # Input colors for trajectory plotting (every 20 degrees).
alpha_3d = 1  # Transparency for 3D trajectories.
alpha_proj = 0.1  # Transparency for 2D projections.
data_out_dir = './figs/fig_data/manifold.json'
proj_z_value={'stim1': -8, 'interval': -7, 'go_cue': -4, 'response': -4, 'all': -12}
sigma_rec, sigma_x = 0, 0  # No noise.

if gen_data:
    sub = Agent(model_dir + sub_dir, rule_name)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

    sub_dic = {'epochs': sub.epochs, 'fir_rate': sub.fir_rate, 'target_color': sub.behaviour['target_color']}

    tools.save_dic(sub_dic, data_out_dir)

# Initialize the manifold plotter
sub = tools.load_dic(data_out_dir)
mplot = MPloter()
mplot.load_data(sub['fir_rate'], sub['epochs'], sub['target_color'])

# Perform PCA analysis on the neural activity (3 components)
mplot._pca_fit(3, start_time=sub['epochs']['stim1'][0] - 1, end_time=sub['epochs']['response'][1])
#plot_range = [[-12, 12], [-12, 12], [-8, 8]] # the limits of x, y, and z axis

for epoch_name in ['stim1', 'interval', 'go_cue', 'response', 'all']:
    fig = plt.figure(figsize=(3, 3))
    axext = fig.add_subplot(111, projection='3d')

    # Plot trajectories for the specific epoch
    if epoch_name == 'all':
        _, ax = mplot.pca_3d_plot(start_time=sub['epochs']['stim1'][0] - 1, end_time=sub['epochs']['response'][1], ax = axext, proj_z_value = proj_z_value[epoch_name], alpha_3d=alpha_3d, alpha_proj=alpha_proj, do_pca_fit=False) # -1 because the python do not read the last time point. For example, the fix period are [0, 1, 2, 3, 4]. In the 5th time point, stim1 has already start. We really wanna to also include the last time point in the last period for comparison
    else:
        _, ax = mplot.pca_3d_plot(start_time=sub['epochs'][epoch_name][0] - 1, end_time=sub['epochs'][epoch_name][1], ax = axext, proj_z_value = proj_z_value[epoch_name], alpha_3d=alpha_3d, alpha_proj=alpha_proj, do_pca_fit=False)

    if epoch_name == 'stim1':
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    #ax.set_xlim(plot_range[0])
    #ax.set_ylim(plot_range[1])
    #ax.set_zlim(plot_range[2])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    fig.tight_layout()
    fig.savefig('./figs/fig_collect/manifold_' + epoch_name + '.pdf', format='pdf')
    #plt.show()

# Visualize an additional projection at the end of the interval epoch
fig = plt.figure(figsize = (5, 3))
axext = fig.add_subplot(111, projection='3d')
_, ax = mplot.pca_3d_plot(start_time=sub['epochs']['interval'][1], end_time=sub['epochs']['interval'][1] + 1, ax = axext, proj_z_value = -3, alpha_3d=alpha_3d, alpha_proj=alpha_proj, do_pca_fit=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()
