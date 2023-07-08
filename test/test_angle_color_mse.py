# compute the variance of angle and color in one example trial
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
prior_sig = 15.0 # width of the piror
sigma_rec = None; sigma_x = None # set the noise to be default (training value)

rule_name = 'color_reproduction_delay_unit'
model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
model_dir = 'model_1/' # example RNN
sub_dir = 'noise_delta/'

f = os.path.join(model_dir_parent, model_dir, sub_dir)
sub = Agent(f, rule_name)

sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=[input_color]) # one example trial
example_trial_state = sub.state.copy() # save one example trial. The shape is (n_time, n_trial=1, 256)
print(example_trial_state.shape)

sub.do_batch_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=300) # a lot of trials only for fitting the pc1-pc2 plane, see sa.angle(fit_pca=True)
sa = State_analyzer()
sa.read_rnn_agent(sub)
# compute the angle(t)
angle = sa.angle(example_trial_state[:, 0], fit_pca=True, state_type='data')
color = sa.angle_color(angle, input_var='angle')

plt.figure()
time = np.arange(angle.shape[0])
plt.scatter(time, angle)
plt.plot(time, angle)
phi = 280 # there's a phase difference between angle and color. Use optimization algorithm to find this phase. Then we can match angle and color to the same figure
plt.scatter(time, color)
plt.plot(time, color)
plt.show()

print(np.var(angle[10:])) # you may wanna consider removing the early time points and outliers (which affects variance a lot)
print(np.var(color[10:]))

# visualize state on the pc1-pc2 background see figs/combine_state.py for more detail

