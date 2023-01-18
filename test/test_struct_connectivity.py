# This file calculate the structure connectivity of an RNN.
# 1. Do experiments with different input colors
# 2. Collect firing rates, label neurons with their preferred color
# 3. Obtain weight matrix of the RNN
# 4. If bin_width is not None, the result weight_pped is the averaged structure connectivity. It is a matrix of shape (n_color, n_color), each element is the connection strength from neuron preferred to color 1 to neurons preferred to color 2. For example, the RNN may have 10 neurons prefer color 100, 3 neurons prefer color 5, then the connectivity from color 100 to color 5 is the averge of 3 * 100 connections.

import context
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir, sc_dist

model_dir = '../core/model/model_10.0/color_reproduction_delay_unit/model_0/' # one rnn model
rule_name = 'color_reproduction_delay_unit' # rule name is fixed to color_reproduction_delay_unit. Actually this repo can also train another type of RNN with slightly different input format, but in this paper we only use color_reproduction_delay_unit
epoch = 'interval' # delay period
binwidth = 5 # binwidth for output color and target color
batch_size = 1
prod_intervals=800 # set the delay time to 800 ms for ploring the trajectory
pca_degree = np.arange(0, 360, 5) # these would be input colors for experiments, resulting firing rates would be used for calculating tuning curves
tuned_thre = -0.0 # discard neurons with weak color selectivity, for example, a flat tuning curve
bin_width = 8 # connectivity of these neurons with similar preferred color would be averaged

# doing some experiments to collect firing rates
sub = Agent(model_dir, rule_name)
fir_rate_list = []
for i in range(batch_size):
    fir_rate, _, _ = sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0.0, sigma_x=0.0)
    fir_rate_list.append(fir_rate)
# mean firing rate
fir_rate_list = np.concatenate(fir_rate_list).reshape(-1, *fir_rate.shape)
fir_rate_mean = np.mean(fir_rate_list, axis=0)

# get the tunning matrix
bump = Bump_activity()
bump.fit(sub.behaviour['target_color'], fir_rate_mean, sub.epochs['interval'])

# average weight by neurons prefered color
weight_hh = sub.model.weight_hh.detach().cpu().numpy()
weight_pped, label = sc_dist(bump, weight_hh, thre=tuned_thre, bin_width=bin_width)
plt.imshow(weight_pped, cmap='seismic')
plt.show()
