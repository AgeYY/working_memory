import context
import numpy as np
import os
from core.manifold.fix_point import Hidden0_helper
from core.agent import Agent
from core.rnn_decoder import RNN_decoder
import hickle as hkl
#from core.state_evolver import evolve_recurrent_state
from core.state_evolver import State_Evolver
from core.tools import state_to_angle
import matplotlib.pyplot as plt
from scipy.stats import entropy
from brokenaxes import brokenaxes
import math

plt.rc('ytick', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('axes', linewidth=2)

# Directories
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
# period_name = 'interval' # can be response, of interval PC1-PC2 plane
# evolve_period = ['go_cue', 'go_cue']

# to study the drift of response period, use
period_name = 'response' # can be response, of interval PC1-PC2 plane
evolve_period = ['response', 'response']

entropy_start_all = []
entropy_end_all = []

for prior_sig in sigmas:
# file names
    adapted_model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
    entropy_start_sig = []
    entropy_end_sig = []

    for i in range(50):
        model_dir = 'model_'+str(i)+'/'  # example RNN
        model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)

        ##### Get mesh points in the response PC1-PC2 plane
        sub = Agent(model_file, rule_name)
        sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x) # generate initial states

        hhelper = Hidden0_helper(hidden_size=256)
        #cords_pca, cords_origin = hhelper.mesh_pca_plane(sub, period_name=period_name, xlim=xlim, ylim=ylim, edge_batch_size=edge_batch_size) # get the
        cords_pca, cords_origin = hhelper.delay_ring(sub, period_name=period_name)

        se = State_Evolver()
        se.read_rnn_file(model_file, rule_name)
        se.set_trial_para(prod_interval=prod_intervals)
        states = se.evolve(cords_origin, evolve_period=evolve_period)
        # print(states.shape)
        angle_s = state_to_angle(states[0])
        if evolve_period[1] == 'go_cue':
            angle_e = state_to_angle(states[-1])
        elif evolve_period[1] == 'response':
            angle_e = state_to_angle(np.mean(states, axis=0))

        hist_s, _ = np.histogram(angle_s, bins=bin_edges, density=True)
        hist_e, _ = np.histogram(angle_e, bins=bin_edges, density=True)

        entropy_s = entropy(hist_s)
        entropy_e = entropy(hist_e)

        entropy_start_sig.append(entropy_s)
        entropy_end_sig.append(entropy_e)

        print('Sigma_s={s}, model {i}: entropy_s = {es}, entropy_end = {ee}'.format(s=prior_sig, i=i, es=entropy_s,ee=entropy_e))

    entropy_start_all.append(entropy_start_sig)
    entropy_end_all.append(entropy_end_sig)

# TEMP: if you like saving data
# data = {'entropy_start_all': entropy_start_all, 'entropy_end_all': entropy_end_all, 'sigmas': sigmas}
# hkl.dump(data, './figs/fig_data/entropy_res_start_end_all.hkl')

# evolve_period = ['go_cue', 'go_cue']
# evolve_period = ['response', 'response']
# data_go = hkl.load('./figs/fig_data/entropy_res_start_end_all.hkl')
# entropy_start_all, entropy_end_all  = data_go['entropy_start_all'], data_go['entropy_end_all']

entropy_start_all = np.array(entropy_start_all)
entropy_end_all = np.array(entropy_end_all)

entropy_start_mean = list(np.mean(entropy_start_all,axis=1))
# entropy_start_ste = list(np.std(entropy_start_all, axis=1) / math.sqrt(entropy_start_all.shape[1]))
entropy_start_std = list(np.std(entropy_start_all, axis=1) )
print(entropy_start_std)
entropy_end_mean = list(np.mean(entropy_end_all,axis=1))
# entropy_end_ste = list(np.std(entropy_end_all, axis=1) / math.sqrt(entropy_end_all.shape[1]))
entropy_end_std = list(np.std(entropy_end_all, axis=1))
print(entropy_end_std)


########

fig = plt.figure(figsize=(3,3))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

# bax.plot(sigmas,unadapted_theo_means,'k',linestyle='--',alpha=0.5)
bax.errorbar(x=sigmas, y=entropy_start_mean,yerr=entropy_start_std,fmt='b.-',linewidth=1.5, markersize=8,label='Start',alpha=1)
bax.errorbar(x=sigmas, y=entropy_end_mean,yerr=entropy_end_std,fmt='r.-',linewidth=1.5, markersize=8,label='End',alpha=1)

bax.set_ylabel('Entropy',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.legend(loc='lower right', frameon=False)
plt.savefig('./figs/fig_collect/drift_entropy_'+evolve_period[0]+'_sigmas.svg',format='svg',bbox_inches='tight')
plt.show()
