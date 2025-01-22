# report distribution
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

# Attempt to parse command-line arguments for model directory, rule name, and sub-directory.
try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
# Default values if command-line arguments are not provided.
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model/model_25.0/color_reproduction_delay_unit/'
    sub_dir = '/noise_delta'

# Determine whether to generate new data or use existing data.
try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False


# Configuration parameters
binwidth = 3  # Bin width for histogram plots.
n_bins = 360 // binwidth  # Total number of bins for plotting.
prod_intervals=1000  # Delay interval for data generation (1,000 ms).
pca_degree = np.arange(0, 360, 5)   # Degrees of colors to plot trajectories.
common_color = [40, 130, 220, 310]  # Common colors used in the training prior.
batch_size = 1000  # Number of trials.
out_path = './figs/fig_data/report_dist.csv'
sigma_rec = None
fs = 12

def gen_report_target(out_path):
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
    group.do_batch_exp(prod_intervals=prod_intervals, sigma_rec=sigma_rec, batch_size=batch_size)

    dire = group.group_behaviour.copy()  # Group behavior data.
    dire_df = pd.DataFrame(dire)
    dire_df.to_csv(out_path)

if gen_data:
    gen_report_target(out_path)


sns.set_theme(style='ticks')


# Function to plot the distribution of output colors
def plot_report_dist(report_dist, add_common_color=False):
    '''
    report_dist (array [float]): report colors
    add_common_color (bool): add 4 dash lines to indicate the position of common color

    '''
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
    hist, bin_edges = np.histogram(report_dist, bins=n_bins, density=True) # distribution density of degs_samples. Totally len(bins) - 1 bins.

    color = Degree_color()
    RGBA = color.out_color(bin_edges, fmat='RGBA')

    for i in range(n_bins):
        #ax.fill_between(bin_edges[i:i+2], hist[i:i+2], hist[i:i+2] - 0.0005, color=RGBA[i])
        ax.fill_between(bin_edges[i:i+2], hist[i:i+2], color=RGBA[i])

    if add_common_color:
        for cc_i in common_color:
            ax.axvline(x = cc_i, linestyle = '--', linewidth = 3, color = 'black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim([0, 360])
    ax.set_ylim([0, 4.5e-3])

    ax.set_xticks([0, 360])
    ax.set_yticks([4e-3])

    ax.tick_params(direction='in')

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    return fig, ax


if __name__ == "__main__":
    ### plot RNN result
    dire_df = pd.read_csv(out_path)

    # Plot the report distribution with common color indicators.
    fig, ax = plot_report_dist(dire_df['report_color'], add_common_color=True)
    ax.set_xlabel('Response value', fontsize=fs)
    fig.savefig('./figs/fig_collect/report_rnn.pdf', format='pdf')
    #plt.show()
