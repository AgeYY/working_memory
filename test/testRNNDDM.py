# plot the information trajectories of RNN model during the delay period
import context
from core.agent import Agent
import numpy as np
from core.diff_drift import Diff_Drift
import matplotlib.pyplot as plt

model_dir = '../core/model_local/color_reproduction_delay_unit_vonmise_cp7_np2/model_30/noise_delta_p2'
rule_name = 'color_reproduction_delay_unit'

sub = Agent(model_dir, rule_name)
ddf = Diff_Drift()
ddf.read_rnn_agent(sub)

time, color = ddf.traj(n_colors=30, sigma_x=None, sigma_rec=None, padding=0, prod_intervals=2000)

plt.plot(time, color)
plt.xlabel('Time (ms)')
plt.ylabel('Angle (degree)')
plt.show()
