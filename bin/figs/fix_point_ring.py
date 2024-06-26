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

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_90.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/model_3/noise_delta", type=str,
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

out_path = './figs/fig_data/fixpoints_ring_' + file_label + '.json'
fig_out_path = './figs/fig_collect/traj_fix_' + file_label


hidden_size = 256
prod_interval_mplot = prod_intervals # 800 for ploting delay trajectories, not for searching fixpoints. 5000 for long delay epoch
prod_interval_search = 300 # for initialize the searching points for find_fixpoints
batch_size = 300
n_epochs = 10000
lr=1
speed_thre = None # speed lower than this we consider it as fixpoints, slow points otherwise
milestones = [3000, 6000, 9000]
alpha=0.7
initial_type='delay_ring'
sigma_init = 0 # Specify the noise adding on initial searching points

if gen_data:
    fixpoint_output = ultimate_find_fixpoints(model_dir + sub_dir, rule_name, batch_size=batch_size, n_epochs=n_epochs, lr=lr, speed_thre=speed_thre, milestones=milestones, initial_type=initial_type, sigma_init=sigma_init, prod_intervals=prod_interval_search, min_angle=5, n_epoch_clect_slow_points=100)
    tools.save_dic(fixpoint_output, out_path)

fp_out = tools.load_dic(out_path)
fixpoints = fp_out['fixpoints']
fixpoints = np.array(fixpoints)

#################### Plot
pca_degree = np.linspace(0, 360, 100, endpoint=False) # Plot the trajectories of these colors
sub = Agent(model_dir+sub_dir, rule_name)
sa = State_analyzer(prod_intervals=800, pca_degree=pca_degree, sigma_rec=0, sigma_x=0)
sa.read_rnn_agent(sub)

##### Plot the speed distribution
vel = sa.velocity_state(fixpoints)
speed = np.linalg.norm(vel, axis=1)

angle = np.array(fp_out['angle'])
#fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
#ax.scatter(angle / 360 * 2 * np.pi, -np.log10(speed))
#
#fig.savefig('./figs/fig_collect/fix_point_speed.pdf', format='pdf')

##### Plot delay trajectories and the fixpoints
sub.do_exp(prod_intervals=prod_interval_mplot, ring_centers=np.linspace(0, 360, 30, endpoint=False), sigma_rec=0, sigma_x=0) # used to plot backgroud trajectories
mplot = MPloter()
mplot.load_data(sub.state, sub.epochs, sub.behaviour['target_color'])
mplot._pca_fit(2, start_time=sub.epochs['interval'][0] - 1, end_time=sub.epochs['interval'][1])
fig_2d = plt.figure(figsize=(3, 3))
axext_2d = fig_2d.add_subplot(111)

_, ax = mplot.pca_2d_plot(start_time=sub.epochs['interval'][0], end_time=sub.epochs['interval'][1], ax = axext_2d, alpha=alpha, do_pca_fit=False, end_point_size=100)

fixpoints_2d = mplot.pca.transform(fixpoints)

att_status = np.array(fp_out['att_status'], dtype=bool)

axext_2d.scatter(fixpoints_2d[att_status, 0], fixpoints_2d[att_status, 1], color='black', s=50)
saddle_status = np.logical_not(fp_out['att_status'])
axext_2d.scatter(fixpoints_2d[saddle_status, 0], fixpoints_2d[saddle_status, 1], color='black', marker='+', s=50)
plt.axis('off')


########## Plot unstable modes
def plot_vec(vec, i): # plot a eigenvector on the 2d plane
    end_pts = np.array([+vec, -vec]) * 4
    end_pts_2d = mplot.pca.transform([fixpoints[i], fixpoints[i]] + end_pts)
    axext_2d.plot(end_pts_2d[:, 0], end_pts_2d[:, 1], color='black')

jacs = np.array(fp_out['jacs'], dtype=float)
for i, jac in enumerate(jacs):
    if not att_status[i]: # if it is a saddle points
        eigval, eigvec = sa.eigen(np.array([jac]))

        for j, egv in enumerate(eigval[0]): # all eigenvalues for this saddle points
            if np.real(egv) > 0: # unstable mode
                vec = np.real(eigvec[0, :, j])
                plot_vec(vec, i)


fig_2d.savefig(fig_out_path + "_delay.svg", format='svg')

fig_stim = plt.figure(figsize=(3, 3))
ax_stim = fig_stim.add_subplot(111)
_, ax_stim = mplot.pca_2d_plot(start_time=sub.epochs['stim1'][0] - 1, end_time=sub.epochs['stim1'][1], ax = ax_stim, alpha=alpha, do_pca_fit=True, end_point_size=100)
plt.axis('off')
fig_stim.savefig(fig_out_path + "_stim.svg", format='svg')

#plt.show()
