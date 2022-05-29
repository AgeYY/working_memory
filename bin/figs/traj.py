import context
import sys
from core.rnn_decoder import RNN_decoder
from core.agent import Agent
import numpy as np
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
import matplotlib.pyplot as plt
from core.color_manager import Degree_color
from sklearn.decomposition import PCA
import core.tools as tools
from core.data_plot.plot_tool import color_curve_plot
import torch
from core.manifold.state_analyzer import State_analyzer

fig_out_path = './figs/fig_collect/traj'

model_label = ["90", "10", "25", "3"]
model_dir_list = ["../core/model/color_reproduction_delay_unit_90/",
                  "../core/model/color_reproduction_delay_unit_25/",
                  "../core/model/color_reproduction_delay_unit_3/",]
sub_dir_list = ["/model_7/noise_delta",
                "/model_6/noise_delta",
                "/model_0/noise_delta",]

rule_name="color_reproduction_delay_unit"
n_trial = 30
prod_intervals_mplot = 800 # for ploting delay trajectories, not for searching fixpoints
alpha=0.7

pca_degree = np.linspace(0, 360, n_trial, endpoint=False) # Plot the trajectories of these colors

def plot_traj(mplot, start_time, end_time, ax):
    mplot._pca_fit(2, start_time=sub.epochs['interval'][0], end_time=sub.epochs['interval'][1])
    _, ax = mplot.pca_2d_plot(start_time=start_time, end_time=end_time, ax = ax, alpha=alpha, do_pca_fit=False)

    plt.axis('off')
    return ax

n_sub = len(model_dir_list)
for i in range(n_sub):
    sub = Agent(model_dir_list[i]+sub_dir_list[i], rule_name)

    ##### Plot delay trajectories and the fixpoints
    sub.do_exp(prod_intervals=800, ring_centers=pca_degree, sigma_rec=0, sigma_x=0) # used to plot backgroud trajectories

    mplot = MPloter()
    mplot.load_data(sub.state, sub.epochs, sub.behaviour['target_color'])

    ## stim trajectory
    fig_stim = plt.figure(figsize=(3, 3))
    ax_stim = fig_stim.add_subplot(111)
    ax_stim = plot_traj(mplot, sub.epochs['stim1'][0], sub.epochs['stim1'][1], ax_stim)
    fig_stim.suptitle('Perception Period')
    fig_out_path_temp = fig_out_path + "_stim" + "_" + model_label[i] + ".pdf"
    fig_stim.savefig(fig_out_path_temp, format='pdf')

    ## stim trajectory
    fig_delay = plt.figure(figsize=(3, 3))
    ax_delay = fig_delay.add_subplot(111)
    ax_delay = plot_traj(mplot, sub.epochs['interval'][0], sub.epochs['interval'][1], ax_delay)
    fig_delay.suptitle('Delay Period')
    fig_out_path_temp = fig_out_path + "_delay" + "_" + model_label[i] + ".pdf"
    fig_delay.savefig(fig_out_path_temp, format='pdf')

