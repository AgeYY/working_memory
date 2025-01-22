import context
import numpy as np
import pickle
import os
from core.tools import removeOutliers
from core.manifold.fix_point import Hidden0_helper
from core.agent import Agent
from core.state_evolver import State_Evolver
from core.tools import state_to_angle
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import math
from core.post_delay_metric_analysis import (
    compute_metric, setup_plotting_style, 
    create_broken_axis_plot, process_metric_data
)

# Set up plotting style
setup_plotting_style()

# Directories
metric_name = 'cv'  # can be 'entropy' or 'cv' (coefficient of variation)
rule_name = 'color_reproduction_delay_unit'
sub_dir = 'noise_delta/'
model_names = ['3.0','10.0','12.5','15.0','17.5','20.0', '22.5','25.0','27.5', '30.0','90.0']
sigmas = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]  # Environmental prior

# Paramters to get appropriate neural states
prod_intervals = 800  # Delay duration
sigma_rec, sigma_x = 0, 0  # Noise, set to zero
n_colors = 20  # Number of colors
pca_degree = np.linspace(0, 360, n_colors, endpoint=False)  # Degree of colors

bin_edges = list(np.arange(0, 361, 10)) # 10-degree bins for histograms

# parameters about sampling grids on the PC1-PC2 plane
xlim = [-30, 30]
ylim = [-30, 30]
edge_batch_size = 50 # number of points in each direction

# # to study the drift of go period, use
# period_name = 'interval' # can be response, of interval PC1-PC2 plane
# evolve_period = ['go_cue', 'go_cue']

# to study the drift of response period, use
period_name = 'response' # can be response, of interval PC1-PC2 plane
evolve_period = ['response', 'response']

######## Metric calculation
metric_start_all = []
metric_end_all = []

for prior_sig in sigmas:
    # file names
    adapted_model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
    metric_start_sig = []
    metric_end_sig = []

    # Loop over 50 RNNs for each sigma
    for i in range(50):
        model_dir = 'model_'+str(i)+'/'  # example RNN
        model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)

        ##### Get mesh points in the response PC1-PC2 plane
        sub = Agent(model_file, rule_name)
        sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

        #### Sample grid points in the PCA plane
        hhelper = Hidden0_helper(hidden_size=256)
        cords_pca, cords_origin = hhelper.delay_ring(sub, period_name=period_name)

        #### Evolve neural states through the specified period
        se = State_Evolver()
        se.read_rnn_file(model_file, rule_name)
        se.set_trial_para(prod_interval=prod_intervals)
        states = se.evolve(cords_origin, evolve_period=evolve_period)

        #### Convert neural states to angles
        angle_s = state_to_angle(states[0])
        if evolve_period[1] == 'go_cue':
            angle_e = state_to_angle(states[-1])
        elif evolve_period[1] == 'response':
            angle_e = state_to_angle(np.mean(states, axis=0))

        #### Compute metric for start and end states
        metric_s = compute_metric(angle_s, metric_type=metric_name, bins=bin_edges)
        metric_e = compute_metric(angle_e, metric_type=metric_name, bins=bin_edges)

        metric_start_sig.append(metric_s)
        metric_end_sig.append(metric_e)

        print('Sigma_s={s}, model {i}: metric_s = {es}, metric_end = {ee}'.format(
            s=prior_sig, i=i, es=metric_s, ee=metric_e))

    metric_start_all.append(metric_start_sig)
    metric_end_all.append(metric_end_sig)

metric_start_all = np.array(metric_start_all)
metric_end_all = np.array(metric_end_all)

# Save results
with open('./figs/fig_data/drift_' + metric_name + '_' + evolve_period[0] + '_sigmas.txt', 'wb') as fp:
    pickle.dump((metric_start_all, metric_end_all), fp)

# Load data
with open('./figs/fig_data/drift_' + metric_name + '_' + evolve_period[0] + '_sigmas.txt', 'rb') as fp:
    metric_start_all, metric_end_all = pickle.load(fp)

######## Plot the figure
metric_start_mean, metric_start_std = process_metric_data(metric_start_all, error_type='std')
metric_end_mean, metric_end_std = process_metric_data(metric_end_all, error_type='std')

# Create plot for start states
fig = plt.figure(figsize=(3,3))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

bax.errorbar(x=sigmas, y=metric_end_mean, yerr=metric_end_std, 
             fmt='k.-', linewidth=1.5, markersize=8, label='End', alpha=1)
bax.axhline(y=metric_end_mean[-1], color='k', linestyle='--', alpha=0.5)

# Add shaded error band
bax.fill_between(sigmas, metric_end_mean[-1] - metric_end_std[-1], metric_end_mean[-1] + metric_end_std[-1],
                 color='gray', alpha=0.2)

if metric_name == 'entropy':
    ylabel = 'Entropy'
else:
    if evolve_period[0] == 'go_cue':
        ylabel = 'Coefficient of Variation (CV) of go-end neural states against angles'
    else:
        ylabel = 'CV of representative neural states against angles'
bax.set_ylabel(ylabel, fontsize=13)
bax.set_xlabel(r'$\sigma_s$', fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.legend(loc='lower right', frameon=False)
plt.savefig('figs/fig_collect/drift_' + metric_name + '_' + evolve_period[0] + '_sigmas.svg',
            format='svg', bbox_inches='tight')
plt.show()
