import context
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir, sc_dist

model_dir = '../core/model/color_reproduction_delay_tri/model_0/'

rule_name = 'color_reproduction_delay_tri'
epoch = 'interval'
binwidth = 5 # binwidth for outdire and target dire
batch_size = 1
prod_intervals=2000 # set the delay time to 800 ms for ploring the trajectory
pca_degree = np.arange(0, 360, 5) # Plot the trajectories of these colors
sigma_rec = None # noise in single trial, not for calculate tunning
sigma_x = None
single_color = 180 # the color for single trial
tuned_thre = -0.0
bin_width = 8

# repeat trials
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

weight_hh = sub.model.weight_hh.detach().cpu().numpy()
weight_pped, label = sc_dist(bump, weight_hh, thre=tuned_thre, bin_width=8)
plt.imshow(weight_pped, cmap='seismic')
plt.show()
