# compute the the color dispersion
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
batch_size = 41
prior_sig = 90.0 # width of the piror
sigma_rec = None; sigma_x = None # set the noise to be default (training value)

rule_name = 'color_reproduction_delay_unit'
model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
model_dir = 'model_0/' # example RNN
sub_dir = 'noise_delta/'
f = os.path.join(model_dir_parent, model_dir, sub_dir)
sub = Agent(f, rule_name) # this is the outside agent creating data

### obtain angle of common color phi_c
sa = State_analyzer()
sa.read_rnn_file(f, rule_name) # I strongly recommand using read_rnn_file instead of creating a agent outside (read_rnn_agent). Agent used within a state_analyzer should not be used outside.

### obtain angle phi_i at the end of delay in repeated trials
input_color_list = np.ones(batch_size) * input_color # repeatly run common color trials
sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_list)
end_of_delay_state = sub.state[sub.epochs['interval'][1]] # shape is [batch_size, hidden_size]
phii = sa.angle(end_of_delay_state, fit_pca=True) # Anyway, remember to fit_pca the first time use angle

### compute the dynamic dispersion
dispersion = np.var(phii) # you may wanna move outlier, use circular norm or something
print(dispersion)


