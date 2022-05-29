import context
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir

model_dir = '../core/model/color_reproduction_delay_unit/model_0_noise/'

rule_name = 'color_reproduction_delay_unit'
epoch = 'interval'
binwidth = 5 # binwidth for outdire and target dire
batch_size = 1
prod_intervals=800 # set the delay time to 800 ms for ploring the trajectory
pca_degree = np.arange(0, 360, 5) # Plot the trajectories of these colors
sigma_rec = None # noise in single trial, not for calculate tunning
sigma_x = None # noise in single trial, not for calculate tunning
single_color = 180 # the color for single trial
tuned_thre = 0.0
bin_width = 8

# repeat trials
sub = Agent(model_dir, rule_name)
fir_rate_list = []
for i in range(batch_size):
    fir_rate, _, _ = sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0.0, sigma_x=0)
    fir_rate_list.append(fir_rate)
# mean firing rate
fir_rate_list = np.concatenate(fir_rate_list).reshape(-1, *fir_rate.shape)
fir_rate_mean = np.mean(fir_rate_list, axis=0)

# get the tunning matrix
bump = Bump_activity()
bump.fit(sub.behaviour['target_color'], fir_rate_mean, sub.epochs['interval'])

tunning_pped, label = bump_pipline(bump, bump.tunning.copy(), thre=tuned_thre, bin_width=bin_width)
plt.imshow(tunning_pped, cmap='binary')
plt.savefig('./tunning.png')
plt.show()

#### Try new expriment
firing_rate, _, _= sub.do_exp(prod_intervals=prod_intervals, ring_centers=np.array(single_color), sigma_rec=sigma_rec, sigma_x=sigma_x)
firing_rate = firing_rate[sub.epochs['interval'][0]+10: sub.epochs['interval'][1]]
firing_rate = firing_rate.reshape((firing_rate.shape[0], firing_rate.shape[2])) # only one color here

firing_rate_pped, label = bump_pipline(bump, firing_rate, thre=tuned_thre, bin_width=bin_width)
plt.imshow(firing_rate_pped, cmap='binary')
plt.savefig('./bump.png')
plt.show()

plt.plot(np.mean(firing_rate_pped[10:15, :], axis=0))
plt.plot(np.mean(firing_rate_pped[-4:, :], axis=0))
plt.savefig('./compare_end.png')
plt.show()
