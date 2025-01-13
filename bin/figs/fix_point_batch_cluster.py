# calculate the distribution of fixpoints. While using mpiexec, number of threads must be a factor of 50
# ERROR: The number of attractors are different for different model. You cannot simply cancatenate threads in this way
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
from mpi4py import MPI

# Initialize MPI for parallel processing
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_90.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/noise_delta", type=str,
                    help='example model')
parser.add_argument('--prod_interval', default=1000, type=int,
                    help='delay epoch length')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be name + file_label.pdf')
parser.add_argument('--gen_data', default=False, action='store_true',
                    help='generate data or not')

arg = parser.parse_args()

# Assign parsed arguments to variables
model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label
gen_data = arg.gen_data

out_path = './figs/fig_data/att_dis_' + file_label + '.json'
fig_out_path = './figs/fig_collect/att_dis_' + file_label + '.svg'

# Parameters for fixed-point finding
batch_size = 300  # Number of trials
n_epochs = 20000  # Maximum number of epochs for training.
lr=1  # Learning rate.
prod_interval_search = 0  # Delay duration for searching fixed points.
speed_thre = None  # Speed threshold to classify fixed points.
milestones = [6000, 12000, 18000]
alpha=0.7
initial_type='delay_ring'   # Type of initial points for fixed-point search.
sigma_init = 0
common_colors = [40, 130, 220, 310]  # Common colors

if gen_data:
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)  # Load a group of RNN agents.
    n_models = len(group.group)
    fixpoint_colors = []  # Store colors of fixed points.
    fixpoint_att = []  # Store attractor status of fixed points.

    for i in range(rank, n_models, size):
        sub = group.group[i]
        model_dir = sub.model_dir

        # Find fixed points for the current model
        fixpoint_output = ultimate_find_fixpoints(model_dir, rule_name, batch_size=batch_size, n_epochs=n_epochs, lr=lr, speed_thre=speed_thre, milestones=milestones, prod_intervals=prod_interval_search, initial_type=initial_type, sigma_init=sigma_init, sub=sub) # find the angle of fixpoints

        # Decode fixed points to color space
        rnn_de = RNN_decoder()
        rnn_de.read_rnn_agent(sub)
        fixpoints = fixpoint_output['fixpoints']
        att = fixpoint_output['att_status']

        # Filter fixed points based on radius (only consider points on the ring)
        pca_degree = np.linspace(0, 360, 100, endpoint=False)  # Input color range.
        sa = State_analyzer(prod_intervals=800, pca_degree=pca_degree, sigma_rec=0, sigma_x=0)
        sa.read_rnn_agent(sub)
        fixpoints_proj = sa.projection(fixpoints, fit_pca=True)  # Project fixed points into PCA space.
        norm = np.linalg.norm(fixpoints_proj, axis=1)
        norm_mean = np.mean(norm)
        fixpoints = fixpoints[norm > norm_mean / 2]   # Keep fixed points with sufficient radius.
        att = att[norm > norm_mean / 2]

        fixpoint_att.extend(att)
        colors = rnn_de.decode(fixpoints)  # Decode fixed points into color degrees.
        fixpoint_colors.extend(colors)

    fixpoint_att = np.array(fixpoint_att).flatten().astype(bool)
    fixpoint_colors = np.array(fixpoint_colors).flatten()

    recvbuf_fixpoint_att, recvbuf_fixpoint_color = None, None

    sendcounts_fixpoint_att = np.array(comm.gather(len(fixpoint_att), root=0)) # count the length of array in each thread
    sendcounts_fixpoint_color = np.array(comm.gather(len(fixpoint_colors), root=0)) # count the length of array in each thread

    if rank == 0:
        recvbuf_fixpoint_att = np.empty(np.sum(sendcounts_fixpoint_att), dtype=bool)
        recvbuf_fixpoint_color = np.empty(np.sum(sendcounts_fixpoint_color), dtype=float)
        print(fixpoint_att.shape, recvbuf_fixpoint_att.shape, sendcounts_fixpoint_att)

    comm.Gatherv(sendbuf=fixpoint_att, recvbuf=(recvbuf_fixpoint_att, sendcounts_fixpoint_att), root=0)
    comm.Gatherv(sendbuf=fixpoint_colors, recvbuf=(recvbuf_fixpoint_color, sendcounts_fixpoint_color), root=0)

    if rank == 0: # concatenate
        #fixpoint_att, fixpoint_colors = recvbuf_fixpoint_att.flatten(), recvbuf_fixpoint_color.flatten()
        tools.save_dic({'fixpoint_att': recvbuf_fixpoint_att, 'fixpoint_colors': recvbuf_fixpoint_color}, out_path)

if rank == 0:
    data = tools.load_dic(out_path)
    colors = data['fixpoint_colors']
    fixpoint_att = np.array(data['fixpoint_att'], dtype=bool)
    att_colors = np.array(colors)[fixpoint_att]
    
    sns.set()
    sns.set_style("ticks")

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.25, 0.2, 0.6, 0.6])

    bins = np.histogram_bin_edges(att_colors, bins=18, range=(0, 360))

    sns.histplot(att_colors, ax=ax, bins=bins, stat='probability', color='tab:red')
    for cm in common_colors:
        ax.axvline(x = cm, linestyle = '--', linewidth = 2, color = 'black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Decoded information value')
    ax.set_ylabel('Count of the attractors (normalized)')
    ax.set_xticks([0, 90, 180, 270, 360])

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    fig.savefig(fig_out_path, format='svg')
    plt.show()
