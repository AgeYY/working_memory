import context
import numpy as np
import os
from core.manifold.fix_point import Hidden0_helper
from core.agent import Agent
from core.rnn_decoder import RNN_decoder
#from core.state_evolver import evolve_recurrent_state
from core.state_evolver import State_Evolver
from core.tools import state_to_angle, align_center_multiline
import matplotlib.pyplot as plt
from scipy.stats import entropy

plt.rc('ytick', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('axes', linewidth=2)

#################### Hyperparameters
# Select the model
prior_sig = 3.0
rule_name = 'color_reproduction_delay_unit'
adapted_model_dir_parent = "../core/model_short_res_40/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
model_dir = 'model_5/'  # Example RNN
sub_dir = 'noise_delta/'
model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)

# Paramters to get appropriate neural states
prod_intervals = 100  # Duration of the delay period
sigma_rec, sigma_x = 0, 0  # Noise levels (set to zero to disable noise)
n_colors = 20
pca_degree = np.linspace(0, 360, n_colors, endpoint=False)

# Parameters about sampling grids on the PC1-PC2 plane
xlim = [-30, 30]
ylim = [-30, 30]
edge_batch_size = 50 # number of points in each direction

# to study the drift of go period, use
# period_name = 'interval' # can be response, of interval PC1-PC2 plane
# evolve_period = ['go_cue', 'go_cue']

# to study the drift of response period, use
period_name = 'response' # can be response, of interval PC1-PC2 plane
evolve_period = ['response', 'response']

##### Get mesh points in the response PC1-PC2 plane
sub = Agent(model_file, rule_name)
sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x) # generate initial states

#### Obtain grid points in the PCA plane
hhelper = Hidden0_helper(hidden_size=256)
#cords_pca, cords_origin = hhelper.mesh_pca_plane(sub, period_name=period_name, xlim=xlim, ylim=ylim, edge_batch_size=edge_batch_size) # get the
cords_pca, cords_origin = hhelper.delay_ring(sub, period_name=period_name)

#### Evolve the neural states through the selected period
se = State_Evolver()
se.read_rnn_file(model_file, rule_name)
se.set_trial_para(prod_interval=prod_intervals)
states = se.evolve(cords_origin, evolve_period=evolve_period)
print(states.shape)

#### Convert neural states to angles
angle_s = state_to_angle(states[0])

# If evolving through the go epoch, use the final states
if evolve_period[0] == 'go_cue':
    angle_e = state_to_angle(states[-1])
# If evolving through the response epoch, use the average of the states
elif evolve_period[0] == 'response':
    angle_e = state_to_angle(np.mean(states, axis=0))

hist_s, _ = np.histogram(angle_s, bins=list(np.arange(0, 360, 10)), density=True)
hist_e, _ = np.histogram(angle_e, bins=list(np.arange(0, 360, 10)), density=True)

entropy_s = entropy(hist_s)  # Entropy of the start states
entropy_e = entropy(hist_e)  # Entropy of the end/average states
print(entropy_s, entropy_e)

fig,ax = plt.subplots(figsize=(3.5,3))
ax.hist(angle_s, bins=list(np.arange(0, 360, 10)),density=False,label='Start',alpha=0.7,color='tab:blue')

if evolve_period[0] == 'go_cue':
    end_label = 'End'
elif evolve_period[0] == 'response':
    end_label = 'Average'

ax.hist(angle_e, bins=list(np.arange(0, 360, 10)),density=False,label=end_label,alpha=0.7,color='tab:red')
ax.set_xlim((0,360))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Angle (angle degree)',fontsize=15)
ax.set_ylabel('Number of Neural States',fontsize=15)
ax.legend(frameon=False)
# ax.set_ylim((0,80))
plt.tight_layout()
plt.savefig('../bin/figs/fig_collect/drift_'+evolve_period[0]+'.svg',format='svg',bbox_inches='tight')
plt.show()
