import context
import sys
from core.rnn_decoder import RNN_decoder
from core.agent import Agent
import numpy as np
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
import matplotlib.pyplot as plt
from core.color_manager import Degree_color
from sklearn.decomposition import PCA
import core.tools as tools
from core.data_plot.plot_tool import color_curve_plot
from core.manifold.ultimate_fix_point import ultimate_find_fixpoints
import torch
from core.manifold.state_analyzer import State_analyzer
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

# Add arguments with default values
parser.add_argument('--model_dir', default="../core/model/model_12.5/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/model_3/noise_delta", type=str,
                    help='example model')
parser.add_argument('--prod_interval', default=1000, type=int,
                    help='delay epoch length')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be bumo + file_label.pdf')

arg = parser.parse_args()

# Assign parsed arguments to variables
model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label

gen_data = True

out_path = './figs/fig_data/manifold_2d' + file_label + '.json'
out_fig_path = './figs/fig_collect/' + 'manifold_2d_' + file_label +'_'

# Model-specific parameters
hidden_size = 256  # Size of the hidden layer in the RNN.
prod_intervals_mplot = prod_intervals  # Duration for plotting delay trajectories.
alpha=1
batch_size = 300  # Number of trials

# Load the RNN model
sub = Agent(model_dir+sub_dir, rule_name)

##### Plot delay trajectories
# Run experiments to obtain neural trajectories for delay epoch
sub.do_exp(prod_intervals=prod_intervals_mplot, ring_centers=np.linspace(0, 360, 20, endpoint=False), sigma_rec=0, sigma_x=0) # used to plot backgroud trajectories

# Initialize manifold plotter and load neural activity data
mplot = MPloter()
mplot.load_data(sub.state, sub.epochs, sub.behaviour['target_color'])

# Fit PCA to the neural activity during the delay epoch
# mplot._pca_fit(3, start_time=sub.epochs['stim1'][0] - 1, end_time=sub.epochs['response'][1])
mplot._pca_fit(3, start_time=sub.epochs['interval'][0] - 1, end_time=sub.epochs['interval'][1])

# Function to generate and save 2D manifold visualizations for a given epoch
def manifold_epoch_name(name):
    #mplot._pca_fit(2, start_time=sub.epochs[name][0], end_time=sub.epochs[name][1])
    fig_2d = plt.figure(figsize=(3, 3))
    axext_2d = fig_2d.add_subplot(111)
    _, ax = mplot.pca_2d_plot(start_time=sub.epochs[name][0] - 1, end_time=sub.epochs[name][1], ax = axext_2d, alpha=alpha, do_pca_fit=False)
    plt.axis('off')
    fig_2d.savefig(out_fig_path  + name + '.pdf', format='pdf')

print(sub.epochs)

# Generate and save 2D visualizations for specific epochs
manifold_epoch_name('interval')
manifold_epoch_name('go_cue')
manifold_epoch_name('response')

#plt.show()
