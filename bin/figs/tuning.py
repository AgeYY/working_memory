# tuning matrix, bump activity
import context
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir
import sys
from core.net_struct.main import circular_mean
import pandas as pd
import seaborn as sns
import core.tools as tools

##### input arguments

try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
    label_name = sys.argv[5]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = 'model_16/noise_delta/'
    label_name = ''

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

####################

prod_intervals_tuning = 800 # a parameter used for computing the tuning curve of neurons
pca_degree = np.arange(0, 360, 1) # Plot the trajectories of these colors
sigma_rec, sigma_x = 0, 0
tuned_thre = -999
bin_width = 15

def output_data(sub_dir, model_dir, rule_name):
    # repeat trials
    sub = Agent(model_dir + sub_dir, rule_name)
    fir_rate_list = []
    fir_rate, _, _ = sub.do_exp(prod_intervals=prod_intervals_tuning, ring_centers=pca_degree, sigma_rec=0.0, sigma_x=0.0)
    fir_rate_list.append(fir_rate)
    # mean firing rate
    fir_rate_list = np.concatenate(fir_rate_list).reshape(-1, *fir_rate.shape)
    fir_rate_mean = np.mean(fir_rate_list, axis=0)

    # get the tuning matrix
    bump = Bump_activity()
    bump.fit(sub.behaviour['target_color'], fir_rate_mean, sub.epochs['interval'])

    tuning_pped, label = bump_pipline(bump, bump.tuning.copy(), thre=tuned_thre, bin_width=None)

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
ax.set_ylim([0, 360]) # there is end effect for calculating ci
ax.set_yticks([0, 40, 130, 220, 310, 360])

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.tick_params(bottom=False, left=False)

ax.set_xlabel('Neurons')
ax.set_ylabel('Input color (deg)')

fig.savefig('./figs/fig_collect/tuning_' + label_name + '.pdf', format='pdf')
#plt.show()
