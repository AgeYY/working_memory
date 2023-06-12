# tuning matrix, bump activity
import context
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent
import sys
from core.net_struct.struct_analyzer import Struct_analyzer
import pandas as pd
import seaborn as sns
import core.tools as tools

##### input arguments

try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
    label_name = sys.argv[5]
    gen_data = sys.argv[4]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model/model_3.0/color_reproduction_delay_unit/'
    sub_dir = 'model_0/noise_delta/'
    label_name = ''
    gen_data = 'Y'

try:
    if gen_data == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

####################

prod_intervals_tuning = 800 # a parameter used for computing the tuning curve of neurons
sigma_rec, sigma_x = 0, 0
bin_width_color = 5;
label_method = 'rnn_decoder'
nan_method = 'remove'
n_colors = 1000
generate_state_method = 'delay_ring'
#generate_state_method = 'trial'

def output_data(sub_dir, model_dir, rule_name):
    # repeat trials
    sub = Agent(model_dir + sub_dir, rule_name)
    str_ana = Struct_analyzer()
    str_ana.read_rnn_agent(sub)
    str_ana.prepare_label(n_colors=n_colors, sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=1, prod_intervals=prod_intervals_tuning, method=label_method, bin_width_color=bin_width_color, nan_method=nan_method, generate_state_method=generate_state_method)
    tuning_pped, neuron_label, color_label = str_ana.output_tuning(bin_width_color=bin_width_color, bin_width_neuron=None) # show tuning matrix for each rnn

    ## save data
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
ylim = [0, 360 // bin_width_color]
ax.set_ylim(ylim) # there is end effect for calculating ci
ax.set_yticks(np.linspace(*ylim, 6).astype(int))
a = ax.get_xticks().tolist()
a = np.linspace(0, 360, 6).astype(int)
ax.set_yticklabels(a)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.tick_params(bottom=False, left=False)

ax.set_xlabel('Neurons')
ax.set_ylabel('Input color (deg)')

fig.savefig('./figs/fig_collect/tuning_' + label_name + '.pdf', format='pdf')
plt.show()
