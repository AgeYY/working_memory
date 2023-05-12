# tuning matrix, bump activity
import context
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent
from core.net_struct.bump import bump_finder
import sys
from core.net_struct.struct_analyzer import Struct_analyzer
import pandas as pd
import seaborn as sns
import core.tools as tools
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_10.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/model_0/noise_delta", type=str,
                    help='example model')
parser.add_argument('--prod_interval', default=800, type=int,
                    help='delay epoch length')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be bumo + file_label.pdf')

arg = parser.parse_args()

model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label

gen_data = True

####################

epoch = 'interval' # only show the delay epoch. In this codes, delay epoch is also called intervat
prod_intervals_tuning = 1000 # a parameter used for computing the tuning curve of neurons
single_color = 180 # the color for single trial
tuned_thre = -999
bin_width = 5
diff_start_end = 30 # target differences between the decoded color at the start of delay to the end of the delay
max_single_trial = 2000 # maximum attemps to search bump whose color differences larger than diff_start_end
#bump_sigma_rec, bump_sigma_x = 0, 0
bump_sigma_rec, bump_sigma_x = None, None
generate_state_method = 'delay_ring'
label_method = 'rnn_decoder'

def compute_tuning(sub):
    str_ana = Struct_analyzer()
    str_ana.read_rnn_agent(sub)
    str_ana.prepare_label(sigma_rec=0, sigma_x=0, prod_intervals=prod_intervals_tuning, method=label_method, generate_state_method=generate_state_method)
    tuning_pped, neuron_label, color_label = str_ana.output_tuning(bin_width_color=bin_width, bin_width_neuron=None) # show tuning matrix for each rnn
    return tuning_pped, neuron_label, color_label

def compute_bump(sub):
    bf = bump_finder(input_color=single_color, prod_intervals=prod_intervals, sigma_rec=bump_sigma_rec, sigma_x=bump_sigma_x, delta_color=diff_start_end, max_iter=max_single_trial)
    bf.read_rnn_agent(sub)
    state_list = bf.search_exceed_delta_color()
    bf.prepare_label(sigma_rec=0, sigma_x=0, generate_state_method=generate_state_method)
    fir_rate_pped, label_pped = bf.out_bump(state_list, bin_width=bin_width)
    return fir_rate_pped, label_pped

def output_data(sub_dir, model_dir, rule_name):
    # repeat trials
    sub = Agent(model_dir + sub_dir, rule_name)

    tuning_pped, neural_label, color_label = compute_tuning(sub)
    fir_rate_pped, label_pped = compute_bump(sub)

    ## save data
    pd.DataFrame(fir_rate_pped).to_csv('./figs/fig_data/bump.csv', header=None, index=None)
    pd.DataFrame(label_pped).to_csv('./figs/fig_data/bump_label.csv', header=None, index=None)
    pd.DataFrame(tuning_pped).to_csv('./figs/fig_data/tuning.csv', header=None, index=None)

if gen_data:
    output_data(sub_dir, model_dir, rule_name)

sns.set()
sns.set_style("ticks")

######### Tuning curve
tuning_pped = pd.read_csv('./figs/fig_data/tuning.csv', header=None).to_numpy()

fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
im = ax.imshow(tuning_pped, interpolation='nearest', aspect='auto', cmap='viridis')

fig.colorbar(im)

ax.set_xlim([0, 256]) # there is end effect for calculating ci
ax.set_xticks([0, 256])
ax.set_ylim([0, 360]) # there is end effect for calculating ci
ax.set_yticks([0, 360])

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.tick_params(bottom=False, left=False)

ax.set_xlabel('Neurons')
ax.set_ylabel('Input color (deg)')

fig.savefig('./figs/fig_collect/tuning.pdf', format='pdf')
#plt.show()

firing_rate_pped = pd.read_csv('./figs/fig_data/bump.csv', header=None).to_numpy()
label = pd.read_csv('./figs/fig_data/bump_label.csv', header=None).to_numpy()

from skimage.transform import resize
fr_resized = resize(firing_rate_pped, (60, 60))

fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.25, 0.25, 0.65, 0.6])
#im = ax.imshow(np.exp(alpha_fr * (fr_resized )), interpolation='nearest', aspect='auto', cmap='bwr') # plus 1 to shift the baseline to 0
im = ax.imshow(fr_resized.T, interpolation='nearest', aspect='auto', cmap='bwr') # plus 1 to shift the baseline to 0

ax.set_ylim([0 + 0.5, fr_resized.shape[1] - 0.5]) # there is end effect for calculating ci
ax.set_yticks([0 + 0.5, fr_resized.shape[1]//2, fr_resized.shape[1] - 0.5])
ax.set_yticklabels([0, 180, 360])
ax.set_xlim([0 + 0.5, fr_resized.shape[0] - 0.5]) # there is end effect for calculating ci
n_time = fr_resized.shape[0]
xtick = [0, n_time//4 , n_time//2, n_time * 3 //4, (fr_resized.shape[0] - 0.5)]
ax.set_xticks(xtick)
#ax.set_xticklabels(['0', '250', '500', '750', '1000'])
xticklabel = [str( int(i * prod_intervals/(len(xtick) - 1) ) ) for i in range(len(xtick))]
ax.set_xticklabels(xticklabel)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.tick_params(bottom=False, left=False)

ax.set_ylabel('Neurons \n (Labeled by prefered stimulus)')
ax.set_xlabel('Delay Time (ms)')

cbar = fig.colorbar(im)

cbar.ax.tick_params(direction='out', length=3)

fig.savefig('./figs/fig_collect/bump' + file_label + '.pdf', format='pdf')
#plt.show()
