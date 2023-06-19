# label neural preferred color using geoemtric method
# 1. copute angle of neural axis
# 2. decode color
import os
import context
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from core.agent import Agent
from core.manifold.state_analyzer import State_analyzer
from core.net_struct.struct_analyzer import Struct_analyzer

model_dir_parent = '../core/model/model_17.5/color_reproduction_delay_unit/' # one rnn model
model_dir = model_dir_parent + 'model_0/'
sub_dir = 'noise_delta/'
rule_name = 'color_reproduction_delay_unit' # rule name is fixed to color_reproduction_delay_unit. Actually this repo can also train another type of RNN with slightly different input format, but in this paper we only use color_reproduction_delay_unit

sub = Agent(model_dir + sub_dir, rule_name)

# compute preferred using geometric method (1. compute angle of neural axis 2. compute color)
sa = State_analyzer()
sa.read_rnn_agent(sub)
axis_states = np.identity(256) # (n_recurrent_neuron, n_recurrent_neuron). Each row is a unit vector along an axis of the 256-dimensional space.
angle = sa.angle(axis_states, fit_pca=True, state_type='vector') # (256) angle of 256 neural axes
color_geo = sa.angle_color(angle, input_var='angle') # converting angle to color

# compute preferred color using single neural doctrine, i.e. find the stimuli maximize neural response
str_ana = Struct_analyzer()
str_ana.read_rnn_agent(sub)
str_ana.prepare_label(method='rnn_decoder', bin_width_color=1, generate_state_method='delay_ring', label_neuron_by_mean=False)
tuning_pped, neuron_label, color_label = str_ana.output_tuning(bin_width_color=1, bin_width_neuron=None, sort=False) # draw a delay ring to compute the tuning curve. Each neuron is labeled by the

color_single = neuron_label

plt.figure()
plt.scatter(color_geo, color_single)
plt.show()
