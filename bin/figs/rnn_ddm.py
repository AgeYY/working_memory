import context
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
from core.color_manager import Degree_color
import numpy as np
from core.diff_drift import Diff_Drift, plot_traj
from core.tools import mean_se, save_dic, load_dic, find_nearest
from core.manifold.state_analyzer import State_analyzer
from core.ddm import Euler_Maruyama_solver
import pandas as pd
import sys
from core.ddm import fit_ddm
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_25.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="model_0/noise_delta", type=str,
                    help='example model')
parser.add_argument('--prod_interval', default=1000, type=int,
                    help='delay epoch length')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be name + file_label.pdf')
parser.add_argument('--gen_data', default=True, type=bool,
                    help='generate data or not')

arg = parser.parse_args()

model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label
gen_data = arg.gen_data

out_rnn_dir = './figs/fig_data/rnn_sim_' + file_label + '.json'
out_fig_path = './figs/fig_collect/rnn_ddm_sim_' + file_label + '.pdf'

def find_split_point(arr):
    '''
    if abs(arr[i+1] - arr[i]) > 180, we call point i+1 as split points. 0 is a split point
    output:
      split_points (array)
    '''
    arr_diff = np.abs(np.diff(arr))
    split_points, = np.where(arr_diff > 180)
    split_points = np.insert(split_points + 1, 0, 0).astype(int)
    return split_points


prod_intervals = 1000
n_colors = 200
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
bin_width =10
n_sub = 20
box_space = 20 # if equal to 5 then 0, 5, 10, 15, ..., 360. One box is an interval for example (0, 5). There should be 360 // box_sapce boxes
box_repeat = 20
common_color = [40, 130, 220, 310]
n_bins = 100 # bins for distribution
drift_bin = 5
batch_size_fit_ddm = 300

if gen_data:
    sub = Agent(model_dir + sub_dir, rule_name)
    ddf = Diff_Drift()
    ddf.read_rnn_agent(sub)

    ########## RNN trajectories
    box_id = np.arange(10, 335, box_space) + box_space / 2 # centroid of init_colors
    init_color = np.tile(box_id, box_repeat) # for every centroid we repeat trial multiple times

    rnn_time, rnn_colors = ddf.traj_fix_start(init_color, prod_intervals=prod_intervals, sigma_x=None, sigma_rec=None)
    ########## DDM trajectories
    color_bin, v_bin, noise, noise_loss = fit_ddm(sub, bin_width=2, batch_size=batch_size_fit_ddm, prod_interval=1000, sigma_init=0.05)
    print(noise)
    ems = Euler_Maruyama_solver()
    ems.read_terms(color_bin, v_bin, noise)

    ddm_time, ddm_colors = ems.run(init_color / 360 * 2 * np.pi - np.pi, prod_intervals)
    ddm_colors = (ddm_colors + np.pi) / 2 / np.pi * 360

    save_dic({'rnn_time': rnn_time, 'rnn_colors': rnn_colors, 'ddm_time': ddm_time, 'ddm_colors': ddm_colors}, out_rnn_dir)

def plot_color_traj(time, colors, ax):
    new_color_start = colors[0, :]
    color_deg = Degree_color()
    RGBA = color_deg.out_color(new_color_start, fmat='RGBA')

    for i in range(colors.shape[-1]): # loop over all batches
        split_points = find_split_point(colors[:, i])
        for j in range(len(split_points) - 1):
            ax.plot(rnn_time[split_points[j] : split_points[j+1]], colors[split_points[j] : split_points[j+1], i], color=RGBA[i], linewidth=1)
        ax.plot(time[split_points[-1]:], colors[split_points[-1]:, i], color=RGBA[i], linewidth=1)

    ax.set_ylim([0, 360])
    ax.set_xlim([0, 1000])
    ax.set_yticks([0, 180, 360])
    ax.tick_params(direction='in')

    ax.set_ylabel('Color (degree)')
    ax.set_xlabel('Delay Time (ms)')

data = load_dic(out_rnn_dir)
rnn_time, rnn_colors = np.array(data['rnn_time']), np.array(data['rnn_colors'])
ddm_time, ddm_colors = np.array(data['ddm_time']), np.array(data['ddm_colors'])

fig = plt.figure(figsize=(8, 5))
#fig = plt.figure()
ax_rnn = fig.add_subplot(121)
ax_ddm = fig.add_subplot(122)

plot_color_traj(rnn_time, rnn_colors, ax_rnn)
ax_rnn.set_title('RNN')

plot_color_traj(ddm_time, ddm_colors, ax_ddm)
ax_ddm.set_title('Fitted Diffusion-Drift Model')
ax_ddm.set_yticks([])
ax_ddm.set_ylabel('')

fig.savefig(out_fig_path, format='pdf')

#plt.show()

#################### Plot the report distribution
#def plot_report_dist(report_dist, add_common_color=False):
#    '''
#    report_dist (array [float]): report colors
#    add_common_color (bool): add 4 dash lines to indicate the position of common color
#
#    '''
#    fig = plt.figure(figsize=(4, 3))
#    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
#    hist, bin_edges = np.histogram(report_dist, bins=n_bins, density=True) # distribution density of degs_samples. Totally len(bins) - 1 bins.
#
#    color_deg = Degree_color()
#    RGBA = color_deg.out_color(bin_edges, fmat='RGBA')
#
#    for i in range(n_bins):
#        #ax.fill_between(bin_edges[i:i+2], hist[i:i+2], hist[i:i+2] - 0.0005, color=RGBA[i])
#        ax.fill_between(bin_edges[i:i+2], hist[i:i+2], color=RGBA[i])
#
#    if add_common_color:
#        for cc_i in common_color:
#            ax.axvline(x = cc_i, linestyle = '--', linewidth = 3, color = 'black')
#
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#
#    ax.set_xlim([0, 360])
#    #ax.set_ylim([0, 4.5e-3])
#
#    ax.set_xticks([0, 360])
#    #ax.set_yticks([4e-3])
#
#    ax.tick_params(direction='in')
#
#    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
#    return fig, ax
#
#fig, ax = plot_report_dist(new_color_end, add_common_color=True)
#plt.show()
