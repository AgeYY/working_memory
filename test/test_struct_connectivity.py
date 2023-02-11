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
from core.net_struct.struct_analyzer import Struct_analyzer

model_dir = '../core/model/model_3.0/color_reproduction_delay_unit/model_0/' # one rnn model
rule_name = 'color_reproduction_delay_unit' # rule name is fixed to color_reproduction_delay_unit. Actually this repo can also train another type of RNN with slightly different input format, but in this paper we only use color_reproduction_delay_unit
prod_intervals=800 # set the delay time to 800 ms for ploring the trajectory
n_colors=360 # number of trials, each trial with different color
tuned_thre = -0.0 # discard neurons with weak color selectivity, for example, a flat tuning curve
bin_width_color = 1 # We mesh color bins in calculating tuning curve.
bin_width = 8 # connectivity of these neurons with similar preferred color would be averaged. Note bin_width_color is mesh for tuning curve, but this one is mesh for preferred color.
label_method = 'rnn_decoder' # use rnn decoder to map firing rate with color
nan_method = 'remove' # how to handle nan ==> remove it

# doing some experiments to collect firing rates
sub = Agent(model_dir, rule_name)
str_ana = Struct_analyzer()
str_ana.read_rnn_agent(sub)
str_ana.prepare_label(n_colors=n_colors, sigma_rec=0, sigma_x=0, batch_size=1, prod_intervals=prod_intervals, method=label_method, bin_width_color=bin_width_color, nan_method=nan_method)
weight_hh_pped, label_pped = str_ana.output_weight(thresh=tuned_thre, bin_width=bin_width)

plt.imshow(weight_hh_pped, cmap='seismic')
plt.show()
