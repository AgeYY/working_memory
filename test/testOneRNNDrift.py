# obtain the drift force of RNN
import context
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
import numpy as np
from core.diff_drift import Diff_Drift, plot_traj
from core.tools import mean_se

model_dir = '../core/model_local/color_reproduction_delay_unit_vonmise/model_17/noise_delta_p2'
rule_name = 'color_reproduction_delay_unit'

prod_intervals = 800
n_colors = 100
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
bin_width = 10
n_sub = 9999

sub = Agent(model_dir, rule_name)
ddf = Diff_Drift()
ddf.read_rnn_agent(sub)

time, colors = ddf.traj(padding=0, n_colors=n_colors, sigma_rec=None, sigma_x=None)
plot_traj(time, colors)

#color_bin_group = []
#v_bin_group = []
#
#count = 0
#for sub in group.group:
#    ddf.read_rnn_agent(sub)
#    color_bin, v_bin = ddf.drift(bin_width=bin_width)
#    color_bin_group.append(color_bin)
#    v_bin_group.append(v_bin)
#
#    count=count + 1
#    if count > n_sub:
#        break
#
#color_bin_group = np.concatenate(color_bin_group).flatten()
#v_bin_group = np.concatenate(v_bin_group).flatten()
#
#color, mean_v, se_v = mean_se(color_bin_group, v_bin_group, remove_outlier=True)
#color, mean_v, se_v = color, mean_v / 360 * 2 * np.pi, se_v / 360 * 2 * np.pi # change the velocity unit to rad
#
#fig = plt.figure(figsize=(5, 5))
#ax = fig.add_axes([0.23, 0.2, 0.6, 0.7])
#
#ax.plot(color, mean_v)
#ax.fill_between(color, mean_v - se_v, mean_v + se_v, alpha=0.4)
#plt.show()
