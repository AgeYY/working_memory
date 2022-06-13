import context
import matplotlib.pyplot as plt
from core.color_manager import Color_cell
import numpy as np
from scipy.stats import vonmises
import seaborn as sns
import sys

#################### typical neural activity ####################
from core.agent import Agent
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_25.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/model_0/noise_delta", type=str,
                    help='example model')
parser.add_argument('--prod_interval', default=1000, type=int,
                    help='delay epoch length')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be bumo + file_label.pdf')

arg = parser.parse_args()

model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label

pca_degree = [180] # Plot the trajectories of these colors
sigma_rec, sigma_x = 0, 0 # turn off the noise of RNN
n_neuron = 5

sub = Agent(model_dir + sub_dir, rule_name)
sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)
fir_rate = sub.fir_rate.reshape(-1, 256)
n_time = fir_rate.shape[0]
time_points = np.arange(n_time) * 20
neuron_id = np.arange(n_neuron) * 256 // n_neuron
c_time = [sub.epochs['fix'][1] - 1, sub.epochs['stim1'][1] - 1, sub.epochs['interval'][1] - 1, sub.epochs['go'][1] - 1] # fixation end, perception end, delay end, go cue end
c_time = np.array(c_time) * 20

sns.set_theme()
sns.set_style("ticks")

fig = plt.figure(figsize=(6, 3))
ax = fig.add_axes([0.23, 0.2, 0.6, 0.7])

for n_id in neuron_id:
    ax.plot(time_points, fir_rate[:, n_id])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (a.u.)')

# use dash line to indicate critical time points
for c_t in c_time:
    ax.axvline(x = c_t, linestyle = '--', linewidth = 3, color = 'black')

ax.tick_params(direction='out')
fig.savefig('./figs/fig_collect/neural_activity' + file_label + '.pdf', format='pdf')
#plt.show()
