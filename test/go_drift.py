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

#################### Hyperparameters
# file names
prior_sig = 12.5
rule_name = 'color_reproduction_delay_unit'
adapted_model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
model_dir = 'model_20/'  # example RNN
sub_dir = 'noise_delta/'
model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)

# paramters to get appropriate neural states
prod_intervals = 100
sigma_rec, sigma_x = 0, 0
n_colors = 20
pca_degree = np.linspace(0, 360, n_colors, endpoint=False)

# parameters about sampling grids on the PC1-PC2 plane
xlim = [-30, 30]
ylim = [-30, 30]
edge_batch_size = 50 # number of points in each direction
period_name = 'interval' # can be response, of interval PC1-PC2 plane
#evolve_period = ['go_cue', 'go_cue']
#evolve_period = ['interval', 'interval']
evolve_period = ['response', 'response']

##### Get mesh points in the response PC1-PC2 plane
sub = Agent(model_file, rule_name)
sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x) # generate initial states

hhelper = Hidden0_helper(hidden_size=256)
#cords_pca, cords_origin = hhelper.mesh_pca_plane(sub, period_name=period_name, xlim=xlim, ylim=ylim, edge_batch_size=edge_batch_size) # get the
cords_pca, cords_origin = hhelper.delay_ring(sub, period_name=period_name) # get the

se = State_Evolver()
se.read_rnn_file(model_file, rule_name)
se.set_trial_para(prod_interval=prod_intervals)
states = se.evolve(cords_origin, evolve_period=evolve_period)
print(states.shape)
angle = state_to_angle(states[-1])

plt.hist(angle, bins=20)
plt.show()

###### Decode grid mesh points
#rd = RNN_decoder()
#rd.read_rnn_file(model_file, rule_name)
#
##decode_color = rd.decode(cords_origin, decoding_plane=period_name)
#decode_color = rd.decode(states[0], decoding_plane=period_name)
#
###### quick check the distribution of colors
#plt.hist(decode_color, bins=20)
#plt.show()
