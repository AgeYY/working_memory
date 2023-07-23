# compute teh color density at the common color
import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
from core.manifold.state_analyzer import State_analyzer

prod_int = 800 # duration of the delay
input_color = 40 # the input will be fixed to 40 degree (common color) or 85 degree (uncommon color)
delta = 1 # d color / d phi = ( (color + delta) - (color - delta) ) / ( phi(color + delta) - phi(color - delta) )
prior_sig = 10.0 # width of the piror

rule_name = 'color_reproduction_delay_unit'
model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
model_dir = 'model_1/' # example RNN
sub_dir = 'noise_delta/'
f = os.path.join(model_dir_parent, model_dir, sub_dir)

sa = State_analyzer()
sa.read_rnn_file(f, rule_name) # I strongly recommand using read_rnn_file instead of creating a agent outside (read_rnn_agent). Agent used within a state_analyzer should not be used outside.

phi = sa.angle_color(np.array([input_color - delta, input_color + delta]), input_var='color')
dc_dphi = 2.0 * delta / (phi[1] - phi[0])
print((dc_dphi)**2)
