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

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_25.0/color_reproduction_delay_unit/", type=str,
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

model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label
gen_data = arg.gen_data

out_fig_path = './figs/fig_collect/combine_' + file_label + '.pdf'
out_path = './figs/fig_data/combine_' + file_label + '.json'

hidden_size = 256
n_colors = 200
batch_size = n_colors # batch size for exact ring initial
sigma_rec=0; sigma_x = 0
xlim=[-30, 30]; ylim=[-30, 30]; edge_batch_size=150;
traj_id = int(n_colors / 2.0)
spacing_time = 3 # every two dots in the trajectory would be 20 ms * spacing_time
########## for searching fixpoints
n_epochs = 20000
lr=1
speed_thre = None # speed lower than this we consider it as fixpoints, slow points otherwise
initial_type='delay_ring'
milestones = [6000, 12000, 18000]
batch_size_fp = 500
sigma_init = 0 # Specify the noise adding on initial searching points
prod_interval_search = 100

def gen_data_func():
    ########## Points on the ring
    sub = Agent(model_dir + sub_dir, rule_name)
    pca_degree = np.linspace(0, 360, n_colors, endpoint=False)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)
    
    ##### fit data to find the pca plane
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(sub.state[sub.epochs['interval'][1]])

    ##### state in the hidimensional space and pca plane
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size=hidden_size)
    #hidden0_ring = hhelper.noisy_ring(sub, batch_size=n_colors)
    #hidden0_ring_pca = pca.transform(hidden0_ring)
    hidden0_ring = sub.state[sub.epochs['interval'][0]:sub.epochs['interval'][1], traj_id, :].reshape((-1, 256)) # pick the traj_id trial. Cut the first 100ms
    hidden0_ring_pca = pca.transform(hidden0_ring)

    ##### decode states from high dimesional space
    rnn_de = RNN_decoder()
    rnn_de.read_rnn_agent(sub)

    report_color_ring = rnn_de.decode(hidden0_ring)

    ########## Points on the pca plane
    hidden0_grid_pca, hidden0_grid = hhelper.mesh_pca_plane(sub, xlim, ylim, edge_batch_size)
    hidden0_grid_pca = pca.transform(hidden0_grid)

    report_color_grid = rnn_de.decode(hidden0_grid)

    ########## Find the fixpoints
    fixpoint_output = ultimate_find_fixpoints(model_dir + sub_dir, rule_name, batch_size=batch_size_fp, n_epochs=n_epochs, lr=lr, speed_thre=speed_thre, milestones=milestones, initial_type=initial_type, sigma_init=sigma_init, prod_intervals=prod_interval_search, witheigen=True)

    ###### for ev
    #att_ev_real=[]
    #for i, ev in enumerate(fixpoint_output['eigval']): # iterate every fixpoint
    #    idx_max = np.argmax(ev) # find the largest eigenvalue
    #    if fixpoint_output['att_status'][i]: # if its an attractor
    #        att_ev_real.append(np.real(ev[idx_max])) # collect the eigenvalue

    #print(att_ev_real)
    ###### for ev

    att_status = np.array(fixpoint_output['att_status'], dtype=bool)
    fixpoints = fixpoint_output['fixpoints']
    fixpoints = np.array(fixpoints)
    fixpoints_pca = pca.transform(fixpoints)


    data_dic = {
            'hidden0_grid_pca': hidden0_grid_pca,
            'hidden0_ring_pca': hidden0_ring_pca,
            'report_color_ring': report_color_ring,
            'report_color_grid': report_color_grid,
            'fixpoints_pca': fixpoints_pca,
            'att_status': att_status,
    }

    tools.save_dic(data_dic, out_path)

if gen_data:
    gen_data_func()
########### plot figures

data_df = tools.load_dic(out_path)
hidden0_grid_pca = np.array(data_df['hidden0_grid_pca'])
hidden0_ring_pca = np.array(data_df['hidden0_ring_pca'])
report_color_ring = np.array(data_df['report_color_ring'])
report_color_grid = np.array(data_df['report_color_grid'])
fixpoints_pca = np.array(data_df['fixpoints_pca'])
att_status = np.array(data_df['att_status'], dtype=bool)

deg_color = Degree_color()
colors_ring = deg_color.out_color(report_color_ring, fmat='RGBA')
colors_grid = deg_color.out_color(report_color_grid, fmat='RGBA')

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

ax.scatter(hidden0_grid_pca[:, 0], hidden0_grid_pca[:, 1], c=colors_grid, alpha=1, s=500)

bool_selector = np.arange(hidden0_ring_pca.shape[0])
bool_selector = bool_selector % spacing_time == 0
hidden0_ring_pca_spaced = hidden0_ring_pca[bool_selector]
ax.scatter(hidden0_ring_pca_spaced[:, 0], hidden0_ring_pca_spaced[:, 1], c='grey', alpha=1, s=10)
ax.scatter(hidden0_ring_pca_spaced[0, 0], hidden0_ring_pca_spaced[0, 1], c='grey', alpha=1, s=60, marker='*')
### fix points
ax.scatter(fixpoints_pca[att_status, 0], fixpoints_pca[att_status, 1], color='black', s=80)
#saddle_status = np.logical_not(att_status)
#ax.scatter(fixpoints_pca[saddle_status, 0], fixpoints_pca[saddle_status, 1], color='black', marker='+', s=80)

#color_curve_plot(hidden0_ring_pca[:, 0], hidden0_ring_pca[:, 1], colors='black', ax=ax, kwargs={'linewidth': 3})
#ax.plot(hidden0_ring_pca[:, 0], hidden0_ring_pca[:, 1], c='grey', alpha=0.4, linewidth=8)

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
fig.savefig(out_fig_path, format='pdf')

#plt.show()
