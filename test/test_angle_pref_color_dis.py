# compute the distribution of neural axis angle and preferred color using geometric method
import os
import context
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from core.agent import Agent
from core.manifold.state_analyzer import State_analyzer
from core.tools import state_to_angle

model_dir_parent = '../core/model/model_12.5/color_reproduction_delay_unit/' # one rnn model
rule_name = 'color_reproduction_delay_unit' # rule name is fixed to color_reproduction_delay_unit. Actually this repo can also train another type of RNN with slightly different input format, but in this paper we only use color_reproduction_delay_unit

axis_states = np.identity(256) # (n_recurrent_neuron, n_recurrent_neuron). Each row is a unit vector along an axis of the 256-dimensional space. We will 1. compute angle of each axis to PC0.
angle_list, color_list = [], []
sa = State_analyzer()

count = 0
for filename in os.listdir(model_dir_parent):
    f = os.path.join(model_dir_parent, filename)
    sub = Agent(f, rule_name)
    sa.read_rnn_agent(sub)

    ########## Compute the angle of neural axis to PC0 ##########
    angle, pca = state_to_angle(axis_states, pca=None, state_type='vector', verbose=True) # (256) angle of 256 neural axes. verbose = True to output the fitted pca as well
    angle_list.append(angle)

    ########## Compute the angular ocupation ##########
    '''
    check figs/encode_space.py --> gen_type_RNN for more details

    #####
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x) # do some experiments to obtain the state for later fitting the delay ring. These states will not be used for decoding

    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size=hidden_size)
    hidden0_ring_pca, hidden0_ring = hhelper.delay_ring(sub, batch_size=300) ##### state in the hidimensional space and pca plane
    
    ##### decode states from high dimesional space
    rnn_de = RNN_decoder()
    rnn_de.read_rnn_agent(sub)
    
    report_color_ring = rnn_de.decode(hidden0_ring) # these are the reported colors of the delay ring
    deg = state_to_angle(hidden0_ring, pca=pca, state_type='data', verbose=False) # (n_ring). Compute the angle of the ring states. using report_color_ring and deg one can then compute the angular occupation of color, see encode_space for more detail. The result should be x, mean_y in line 139 of encode_space.py
    x, mean_y, se_y = ...
    '''

# draw the distribution of neural axis angle
angle_list = np.array(angle_list).flatten()

plt.figure()
plt.hist(angle_list, bins=20)

# draw the distribution of angular occupation
'''
...
ax.plot(x, mean_y)
...
'''

########## compute the theoretical prediction of the preferred color distribution
'''
# make a hist for neural axis angle
angle_bin, num_neuron_axis = make a hist for angle_list
color_bin, theo_pref = x, mean_y * num_neuron_axis
'''

# on the other hand, we also need to compute the experimental prediction of the preferred color distribution
