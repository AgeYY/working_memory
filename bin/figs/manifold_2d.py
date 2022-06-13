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

parser.add_argument('--model_dir', default="../core/model/model_25.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/model_0/noise_delta", type=str,
                    help='example model')
parser.add_argument('--prod_interval', default=1000, type=int,
                    help='delay epoch length')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be bumo + file_label.pdf')

arg = parser.parse_args()

model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label

gen_data = True

out_path = './figs/fig_data/manifold_2d' + file_label + '.json'
out_fig_path = './figs/fig_collect/' + 'manifold_2d_' + file_label +'_'

hidden_size = 256
prod_intervals_mplot = prod_intervals # for ploting delay trajectories, not for searching fixpoints
alpha=1
batch_size = 300

sub = Agent(model_dir+sub_dir, rule_name)

#################### Plot
##### Plot delay trajectories
sub.do_exp(prod_intervals=prod_intervals_mplot, ring_centers=np.linspace(0, 360, 20, endpoint=False), sigma_rec=0, sigma_x=0) # used to plot backgroud trajectories
mplot = MPloter()
mplot.load_data(sub.state, sub.epochs, sub.behaviour['target_color'])
#mplot._pca_fit(3, start_time=sub.epochs['stim1'][0] - 1, end_time=sub.epochs['response'][1])
mplot._pca_fit(3, start_time=sub.epochs['interval'][0] - 1, end_time=sub.epochs['interval'][1])

def manifold_epoch_name(name):
    #mplot._pca_fit(2, start_time=sub.epochs[name][0], end_time=sub.epochs[name][1])
    fig_2d = plt.figure(figsize=(3, 3))
    axext_2d = fig_2d.add_subplot(111)
    _, ax = mplot.pca_2d_plot(start_time=sub.epochs[name][0] - 1, end_time=sub.epochs[name][1], ax = axext_2d, alpha=alpha, do_pca_fit=False)
    plt.axis('off')
    fig_2d.savefig(out_fig_path  + name + '.pdf', format='pdf')

print(sub.epochs)
manifold_epoch_name('interval')
manifold_epoch_name('go_cue')
manifold_epoch_name('response')

#plt.show()
