# compute the distribution of neural axis angle and preferred color using geometric method
import os
import context
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from core.agent import Agent
from core.manifold.state_analyzer import State_analyzer

model_dir_parent = '../core/model/model_17.5/color_reproduction_delay_unit/' # one rnn model
model_dir = model_dir_parent + 'model_0/'
rule_name = 'color_reproduction_delay_unit' # rule name is fixed to color_reproduction_delay_unit. Actually this repo can also train another type of RNN with slightly different input format, but in this paper we only use color_reproduction_delay_unit

axis_states = np.identity(256) # (n_recurrent_neuron, n_recurrent_neuron). Each row is a unit vector along an axis of the 256-dimensional space. We will 1. compute angle of each axis to PC0 2. decode the angle into the color
angle_list, color_list = [], []
sa = State_analyzer()

count = 0
for filename in os.listdir(model_dir_parent):
    f = os.path.join(model_dir_parent, filename)
    sub = Agent(f, rule_name)
    sa.read_rnn_agent(sub)

    angle = sa.angle(axis_states, fit_pca=True, state_type='vector') # (256) angle of 256 neural axes
    angle_list.append(angle)

    color = sa.angle_color(angle, input_var='angle') # converting angle to color
    color_list.append(color)

    if count < 60: count += 1
    else: break

angle_list = np.array(angle_list).flatten()
color_list = np.array(color_list).flatten()

plt.figure()
plt.hist(angle_list, bins=20)

plt.figure()
plt.hist(color_list, bins=20)
plt.show()
