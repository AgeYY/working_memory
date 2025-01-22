# same as combine.py, but this draws neural state but not rate
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
from core.data_plot.plot_tool import color_curve_plot
import core.tools as tools
from core.manifold.ultimate_fix_point import ultimate_find_fixpoints
import torch
import argparse
from core.manifold.state_analyzer import State_analyzer
import math

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_12.5/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/model_0/noise_delta", type=str,
                    help='example model')
parser.add_argument('--prod_interval', default=800, type=int,
                    help='delay epoch length')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be bump + file_label.pdf')
parser.add_argument('--gen_data', default=True, type=bool,
                    help='generate figure data')

arg = parser.parse_args()

# Assign parsed arguments to variables
model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label
gen_data = arg.gen_data

out_fig_path = './figs/fig_collect/combine_' + file_label + '.pdf'
out_path = './figs/fig_data/combine_' + file_label + '.json'

# Parameters for visualization
input_color = 40  # The input will be fixed to 40 degree (common color) or 85 degree (uncommon color)
delta = 5  # Angular range for boundary visualization.
hidden_size = 256  # Size of the RNN hidden layer.
n_colors = 200  # Number of colors for decoding.
batch_size = n_colors  # Batch size for exact ring initial
sigma_rec=0; sigma_x = 0  # Noise
xlim=[-30, 30]; ylim=[-30, 30]; edge_batch_size=150;
traj_id = int(n_colors / 2.0)
spacing_time = 3 # every two dots in the trajectory would be 20 ms * spacing_time

def gen_data_func(n_trails=100):
    """
    Generate and save data for visualizing the joint effect of dynamic dispersion
    and angular occupancy in PCA space.
    """

    ########## Calculate color boundary on the PCA plane
    sa = State_analyzer()
    sa.read_rnn_file(model_dir + sub_dir, rule_name)
    phi = sa.angle_color(np.array([input_color - delta, input_color + delta]), input_var='color')
    print(phi)

    ########## Points on the ring
    sub = Agent(model_dir + sub_dir, rule_name)
    pca_degree = np.linspace(0, 360, n_colors, endpoint=False)
    # traj_id = list(pca_degree).index(input_color)
    # print('traj_id',traj_id)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)
    
    ##### Fit PCA to the end of delay epoch activity
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(sub.state[sub.epochs['interval'][1]])

    ##### States in the hidimensional space and pca plane for multiple trials
    hidden0_ring_pca_list = []
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size=hidden_size)
    for _ in range(n_trails):
        sub_2 = Agent(model_dir + sub_dir, rule_name)
        sub_2.do_exp(prod_intervals=prod_intervals, ring_centers=[input_color], sigma_rec=None, sigma_x=None)
        hidden0_ring = sub_2.state[sub_2.epochs['interval'][1]:sub_2.epochs['interval'][1] + 1, 0, :].reshape((-1, 256)) # pick the traj_id trial. Cut the first 100ms
        hidden0_ring_pca = pca.transform(hidden0_ring)
        hidden0_ring_pca_list.append(hidden0_ring_pca)

    ##### Decode states from high dimesional space
    rnn_de = RNN_decoder()
    rnn_de.read_rnn_agent(sub_2)

    report_color_ring = rnn_de.decode(hidden0_ring)

    ########## Points on the pca plane
    hidden0_grid_pca, hidden0_grid = hhelper.mesh_pca_plane(sub, xlim, ylim, edge_batch_size)
    hidden0_grid_pca = pca.transform(hidden0_grid)

    report_color_grid = rnn_de.decode(hidden0_grid)




    data_dic = {
            'hidden0_grid_pca': hidden0_grid_pca,
            'hidden0_ring_pca_list': hidden0_ring_pca_list,
            'report_color_ring': report_color_ring,
            'report_color_grid': report_color_grid,
            'color_boundary': phi
    }

    tools.save_dic(data_dic, out_path)

if gen_data:
    gen_data_func()


########### plot figures
data_df = tools.load_dic(out_path)
hidden0_grid_pca = np.array(data_df['hidden0_grid_pca'])
hidden0_ring_pca_list = np.array(data_df['hidden0_ring_pca_list'])
report_color_ring = np.array(data_df['report_color_ring'])
report_color_grid = np.array(data_df['report_color_grid'])
color_boundary = np.array(data_df['color_boundary'])


# pca_center = [(np.max(hidden0_grid_pca[:,0])+np.min(hidden0_grid_pca[:,0]))/2, (np.max(hidden0_grid_pca[:,1])+np.min(hidden0_grid_pca[:,1]))/2]
pca_center = [0,0]
deg_color = Degree_color()
colors_ring = deg_color.out_color(report_color_ring, fmat='RGBA')
colors_grid = deg_color.out_color(report_color_grid, fmat='RGBA')



fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

ax.scatter(hidden0_grid_pca[:, 0], hidden0_grid_pca[:, 1], c=colors_grid, alpha=1, s=500)

ax.scatter(pca_center[0],pca_center[1], c='k', marker='.', s=5)
for c in [input_color-delta,input_color+delta]:
    k = 30
    dist = np.abs(c - report_color_grid)
    k_nearest = np.argsort(dist)[:k]
    dots = hidden0_grid_pca[k_nearest]
    ax.scatter(dots[:, 0], dots[:, 1], c='k', s=2)

for hidden0_ring_pca in hidden0_ring_pca_list:
    bool_selector = np.arange(hidden0_ring_pca.shape[0])
    bool_selector = bool_selector % spacing_time == 0
    hidden0_ring_pca_spaced = hidden0_ring_pca[bool_selector]
    ax.scatter(hidden0_ring_pca_spaced[-1, 0], hidden0_ring_pca_spaced[-1, 1], c='darkred', marker='o',alpha=0.5, s=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False
)

#fig.savefig('./figs/fig_collect/combine.pdf', format='pdf')
fig.savefig(out_fig_path, format='pdf', dpi=500)

plt.show()
