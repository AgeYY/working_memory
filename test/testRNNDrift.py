# obtain the drift force of RNN
import context
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
import numpy as np
from core.diff_drift import Diff_Drift, plot_traj
from core.tools import mean_se

model_dir = '../core/model/color_reproduction_delay_unit/'
rule_name = 'color_reproduction_delay_unit'

#sub_dir = '/noise_delta_0p25'
sub_dir = '/noise_delta'

prod_intervals = 800
n_colors = 100
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
bin_width = 10
n_sub = 9999

#sub = Agent(model_dir, rule_name)
group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
ddf = Diff_Drift()

color_bin_group = []
v_bin_group = []

count = 0
for sub in group.group:
    ddf.read_rnn_agent(sub)
    color_bin, v_bin = ddf.drift(bin_width=bin_width)
    color_bin_group.append(color_bin)
    v_bin_group.append(v_bin)

    count=count + 1
    if count > n_sub:
        break

color_bin_group = np.concatenate(color_bin_group).flatten()
v_bin_group = np.concatenate(v_bin_group).flatten()

color, mean_v, se_v = mean_se(color_bin_group, v_bin_group, remove_outlier=True)
color, mean_v, se_v = color, mean_v / 360 * 2 * np.pi, se_v / 360 * 2 * np.pi # change the velocity unit to rad

fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.23, 0.2, 0.6, 0.7])

ax.plot(color, mean_v)
ax.fill_between(color, mean_v - se_v, mean_v + se_v, alpha=0.4)
plt.show()

#time, colors = ddf.traj()
#plot_traj(time, colors)
