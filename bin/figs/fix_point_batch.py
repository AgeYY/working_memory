# calculate the distribution of fixpoints
import context
import sys
from core.rnn_decoder import RNN_decoder
from core.agent import Agent, Agent_group
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
import seaborn as sns
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_25.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/noise_delta", type=str,
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

out_path = './figs/fig_data/att_dis_' + file_label + '.json'
fig_out_path = './figs/fig_collect/att_dis_' + file_label + '.pdf'

hidden_size = 256
prod_intervals_mplot = prod_intervals # for ploting delay trajectories, not for searching fixpoints
batch_size = 300
n_epochs = 20000

#batch_size = 3
#n_epochs = 1000

lr=1
prod_interval_search = 100
speed_thre = None # speed lower than this we consider it as fixpoints, slow points otherwise
milestones = [6000, 12000, 18000]
alpha=0.7
initial_type='delay_ring'
sigma_init = 0 # Specify the noise adding on initial searching points
common_colors = [40, 130, 220, 310]

if gen_data:
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)

    fixpoint_colors = []
    fixpoint_att = []

    for sub in group.group:
        model_dir = sub.model_dir
        fixpoint_output = ultimate_find_fixpoints(model_dir, rule_name, batch_size=batch_size, n_epochs=n_epochs, lr=lr, speed_thre=speed_thre, milestones=milestones, prod_intervals=prod_interval_search, initial_type=initial_type, sigma_init=sigma_init) # find the angle of fixpoints
        # decode the angles to color
        rnn_de = RNN_decoder()
        rnn_de.read_rnn_agent(sub)
        fixpoints = fixpoint_output['fixpoints']
        att = fixpoint_output['att_status']

        ## by thresholding the radius of fixpoints, we only consider fixpoints on the ring
        pca_degree = np.linspace(0, 360, 100, endpoint=False) # Plot the trajectories of these colors
        sa = State_analyzer(prod_intervals=800, pca_degree=pca_degree, sigma_rec=0, sigma_x=0)
        sa.read_rnn_agent(sub)
        fixpoints_proj = sa.projection(fixpoints, fit_pca=True)
        norm = np.linalg.norm(fixpoints_proj, axis=1)
        norm_mean = np.mean(norm)
        fixpoints = fixpoints[norm > norm_mean / 2]
        att = att[norm > norm_mean / 2]

        fixpoint_att.extend(att)
        colors = rnn_de.decode(fixpoints)
        fixpoint_colors.extend(colors)

    tools.save_dic({'fixpoint_att': fixpoint_att, 'fixpoint_colors': fixpoint_colors}, out_path)

data = tools.load_dic(out_path)
colors = data['fixpoint_colors']
fixpoint_att = np.array(data['fixpoint_att'], dtype=bool)
att_colors = np.array(colors)[fixpoint_att]

sns.set()
sns.set_style("ticks")

fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.25, 0.2, 0.6, 0.6])

bins = np.histogram_bin_edges(att_colors, bins=18, range=(0, 360))

sns.histplot(att_colors, ax=ax, bins=bins, stat='probability')
for cm in common_colors:
    ax.axvline(x = cm, linestyle = '--', linewidth = 2, color = 'red')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Decoded information value')
ax.set_ylabel('Count of the attractors (normalized)')
ax.set_xticks([0, 90, 180, 270, 360])

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

fig.savefig(fig_out_path, format='pdf')

