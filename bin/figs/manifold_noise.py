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

try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
    label_name = sys.argv[5]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = '/model_16/noise_delta'
    label_name = ''

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

binwidth = 5 # binwidth for outdire and target dire
prod_intervals=500 # set the delay time to 800 ms for ploring the trajectory
pca_degree = np.linspace(0, 360, 15) # Plot the trajectories of these colors
alpha_3d = 1; alpha_proj = 0.2;
data_out_dir = './figs/fig_data/manifold_noise.json'
proj_z_value={'stim1': -8, 'interval': -7, 'go_cue': -8, 'response': -8, 'all': -12}
sigma_rec, sigma_x = None, None

if gen_data:
    sub = Agent(model_dir + sub_dir, rule_name)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

    sub_dic = {'epochs': sub.epochs, 'fir_rate': sub.fir_rate, 'target_color': sub.behaviour['target_color']}

    tools.save_dic(sub_dic, data_out_dir)

sub = tools.load_dic(data_out_dir)
mplot = MPloter()
mplot.load_data(sub['fir_rate'], sub['epochs'], sub['target_color'])
mplot._pca_fit(3, start_time=sub['epochs']['stim1'][0] - 1, end_time=sub['epochs']['response'][1])
#plot_range = [[-12, 12], [-12, 12], [-8, 8]] # the limits of x, y, and z axis

for epoch_name in ['stim1', 'interval', 'go_cue', 'response', 'all']:
    fig = plt.figure(figsize=(5, 3))
    axext = fig.add_subplot(111, projection='3d')

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

    fig.savefig('./figs/fig_collect/manifold_noise_' + epoch_name + '_' + label_name + '.pdf', format='pdf')
    #plt.show()
