import context
import numpy as np
import os
from core.manifold.fix_point import Hidden0_helper
from core.agent import Agent
from core.rnn_decoder import RNN_decoder
#from core.state_evolver import evolve_recurrent_state
from core.state_evolver import State_Evolver
from core.tools import state_to_angle
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import entropy
from brokenaxes import brokenaxes
import math
from core.ploter import plot_layer_boxplot_helper

def SET_MPL_FONT_SIZE(font_size):
    mpl.rcParams['axes.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    return
SET_MPL_FONT_SIZE(13)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['legend.frameon'] = False
# Directories
metric_name = 'cv'
rule_name = 'color_reproduction_delay_unit'
sub_dir = 'noise_delta/'
model_names = ['3.0','10.0','12.5','15.0','17.5','20.0', '22.5','25.0','27.5', '30.0','90.0']
sigmas = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]

# paramters to get appropriate neural states
prod_intervals = 100
sigma_rec, sigma_x = 0, 0
n_colors = 20
pca_degree = np.linspace(0, 360, n_colors, endpoint=False)

bin_edges = bins=list(np.arange(0, 361, 10))

# parameters about sampling grids on the PC1-PC2 plane
xlim = [-30, 30]
ylim = [-30, 30]
edge_batch_size = 50 # number of points in each direction

# to study the drift of go period, use
period_name = 'response' # can be 'interval'
evolve_period = ['response', 'response'] # can be 'go_cue'

def AO_metric(model_dir_root, sigma_s, period_name, evolve_period, t='mean', metric_name='entropy'):
    model_dir_parent = model_dir_root + "/model_" + str(sigma_s) + "/color_reproduction_delay_unit/"
    metric_list = []
    for filename in os.listdir(model_dir_parent):
        f = os.path.join(model_dir_parent, filename)
        ##### Get mesh points in the response PC1-PC2 plane
        sub = Agent(f, rule_name)
        sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec,sigma_x=sigma_x)  # generate initial states

        hhelper = Hidden0_helper(hidden_size=256)
        cords_pca, cords_origin = hhelper.delay_ring(sub, period_name=period_name)

        se = State_Evolver()
        se.read_rnn_file(f, rule_name)
        se.set_trial_para(prod_interval=prod_intervals)
        states = se.evolve(cords_origin, evolve_period=evolve_period)

        if t == 'mean':
            angle = state_to_angle(np.mean(states, axis=0))
        else:
            angle = state_to_angle(states[t])

        hist, _ = np.histogram(angle, bins=bin_edges, density=True)
        dist = hist / np.sum(hist)
        dist_uniform = np.ones(dist.shape) / len(dist)
        # entropy_list.append(entropy(dist) / entropy(dist_uniform))
        if metric_name == 'entropy':
            metric_list.append(entropy(dist))
        elif metric_name == 'cv':
            metric_list.append(np.std(dist) / np.mean(dist))
    return metric_list


######### Compare AO entropy of original model and short-response model (sigma_s = 3.0)
# '''
entropy_ori = AO_metric('../core/model', sigma_s=3.0, period_name=period_name, evolve_period=evolve_period, t='mean', metric_name=metric_name)
entropy_short = AO_metric('../core/model_short_res_40', sigma_s=3.0, period_name=period_name, evolve_period=evolve_period, t='mean', metric_name=metric_name)

score_exps = {'Long': entropy_ori,'Short': entropy_short}
layer_order = {'Long': 0,'Short': 1}
fig, ax = plt.subplots(figsize=(3, 3))
fig, ax = plot_layer_boxplot_helper(score_exps,layer_order, fig=fig, ax=ax, jitter_color='tab:red', jitter_s=30, show_outlier=False)
ax.set_ylabel(metric_name + ' of \n the representative neural states')
fig.tight_layout()
fig.savefig('../bin/figs/fig_collect/long_short_response_'+metric_name+'_'+period_name+'.svg',format='svg',bbox_inches='tight')
plt.show()

# '''

######### Calculte AO entropy for all sigma_s and plot the figure (original model)
'''
entropy_start_all = []
entropy_end_all = []
for prior_sig in sigmas:
    adapted_model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
    entropy_s = AO_entropy("../core/model", prior_sig, period_name, evolve_period, t=0)
    entropy_e = AO_entropy("../core/model", prior_sig, period_name, evolve_period, t=-1)

    print('Sigma_s={s}, mean start entropy: {es}, mean end entropy: {ee}'.format(s=prior_sig, es=np.mean(entropy_s),ee=np.mean(entropy_e)))

    entropy_start_all.append(entropy_s)
    entropy_end_all.append(entropy_e)

entropy_start_all = np.array(entropy_start_all)
entropy_end_all = np.array(entropy_end_all)

entropy_start_mean = list(np.mean(entropy_start_all,axis=1))
# entropy_start_ste = list(np.std(entropy_start_all, axis=1) / math.sqrt(entropy_start_all.shape[1]))
entropy_start_std = list(np.std(entropy_start_all, axis=1) )

entropy_end_mean = list(np.mean(entropy_end_all,axis=1))
# entropy_end_ste = list(np.std(entropy_end_all, axis=1) / math.sqrt(entropy_end_all.shape[1]))
entropy_end_std = list(np.std(entropy_end_all, axis=1))

fig = plt.figure(figsize=(4,3))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

# bax.plot(sigmas,unadapted_theo_means,'k',linestyle='--',alpha=0.5)
bax.errorbar(x=sigmas, y=entropy_start_mean,yerr=entropy_start_std,fmt='b.-',linewidth=1.5, markersize=8,label='Start',alpha=1)
bax.errorbar(x=sigmas, y=entropy_end_mean,yerr=entropy_end_std,fmt='c.-',linewidth=1.5, markersize=8,label='End',alpha=1)

bax.set_ylabel('Entropy',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.legend(loc='lower right')
plt.savefig('../bin/figs/fig_collect/drift_entropy_'+evolve_period[0]+'_sigmas.svg',format='svg',bbox_inches='tight')
plt.show()
# '''


