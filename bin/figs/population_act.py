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
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = 'model_16/noise_delta/'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

####################

epoch = 'interval'
binwidth = 5 # binwidth for outdire and target dire
batch_size = 1
prod_intervals=800 # set the delay time to 800 ms for ploring the trajectory
prod_intervals_tuning = 800
pca_degree = np.arange(0, 360, 1) # Plot the trajectories of these colors
sigma_rec, sigma_x = 0, 0
single_color = 180 # the color for single trial
tuned_thre = -999
bin_width = 15
#diff_start_end = 40
diff_start_end = 0
max_single_trial = 2000
alpha_fr = 5 # increase the bump's contrast by mapping to exp(fr)

def output_data(sub_dir, model_dir, rule_name):
    # repeat trials
    sub = Agent(model_dir + sub_dir, rule_name)
    fir_rate_list = []
    for i in range(batch_size):
        fir_rate, _, _ = sub.do_exp(prod_intervals=prod_intervals_tuning, ring_centers=pca_degree, sigma_rec=0.0, sigma_x=0.0)
        fir_rate_list.append(fir_rate)
    # mean firing rate
    fir_rate_list = np.concatenate(fir_rate_list).reshape(-1, *fir_rate.shape)
    fir_rate_mean = np.mean(fir_rate_list, axis=0)

    # get the tuning matrix
    bump = Bump_activity()
    bump.fit(sub.behaviour['target_color'], fir_rate_mean, sub.epochs['interval'])

    tuning_pped, label = bump_pipline(bump, bump.tuning.copy(), thre=tuned_thre, bin_width=None)

    def do_one_exp(single_color):
        firing_rate, _, _= sub.do_exp(prod_intervals=prod_intervals, ring_centers=np.array(single_color), sigma_rec=sigma_rec, sigma_x=sigma_x)

        firing_rate_delay = firing_rate[sub.epochs['interval'][0]: sub.epochs['interval'][1]]
        firing_rate_delay = firing_rate_delay.reshape((firing_rate_delay.shape[0], firing_rate_delay.shape[2])) # only one color here

        firing_rate = firing_rate.reshape((firing_rate.shape[0], firing_rate.shape[2])) # only one color here
        
        firing_rate_delay_pped, label = bump_pipline(bump, firing_rate, thre=tuned_thre, bin_width=bin_width)
        firing_rate_pped, label = bump_pipline(bump, firing_rate, thre=tuned_thre, bin_width=bin_width)
        return firing_rate_pped, label, sub.behaviour['report_color'][0], single_color

    #### Try new expriment to search bump. We use population vector method to decode the color from the states in delay period. If the difference of decoded color in the begining and end are large, the bump is obvious, we interupt searching.
    for i in range(max_single_trial):
        firing_rate_pped, label, report_color, target_color = do_one_exp(single_color)

        delay_start_fire = np.mean(firing_rate_pped[0:5, :], axis=0)
        delay_end_fire = np.mean(firing_rate_pped[-4:, :], axis=0)  

        delay_start_color, norm = circular_mean(delay_start_fire, label)
        delay_end_color, norm = circular_mean(delay_end_fire, label)

        if abs(delay_start_color - delay_end_color) > diff_start_end:
            print('Searching bump succeed!')
            break

    if i >= max_single_trial:
        print('Searching bump failed!')

    print('delay_start_color: ', delay_start_color)
    print('delay_end_color: ', delay_end_color)

    ## save data
    pd.DataFrame(firing_rate_pped).to_csv('./figs/fig_data/pop_act.csv', header=None, index=None)
    pd.DataFrame(label).to_csv('./figs/fig_data/bump_label.csv', header=None, index=None)
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

firing_rate_pped = pd.read_csv('./figs/fig_data/pop_act.csv', header=None).to_numpy()
label = pd.read_csv('./figs/fig_data/bump_label.csv', header=None).to_numpy()

#delay_start = np.mean(firing_rate_pped[0:5, :], axis=0)
#delay_start_smooth = tools.smooth(delay_start, window_len=8)
#delay_end = np.mean(firing_rate_pped[-4:, :], axis=0)
#delay_end_smooth = tools.smooth(delay_end, window_len=8)
#
##plt.scatter(label, delay_start)
#plt.plot(label, np.exp(delay_start_smooth))
#plt.plot(label, np.exp(delay_end_smooth))
#plt.savefig('./figs/fig_collect/compare_end.pdf', format='pdf')
#plt.show()

from skimage.transform import resize
fr_resized = resize(firing_rate_pped, (60, 60))

fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.25, 0.25, 0.65, 0.6])
#im = ax.imshow(np.exp(alpha_fr * (fr_resized )), interpolation='nearest', aspect='auto', cmap='bwr') # plus 1 to shift the baseline to 0
im = ax.imshow(fr_resized[::-1], interpolation='nearest', aspect='auto', cmap='bwr') # plus 1 to shift the baseline to 0

ax.set_xlim([0 + 0.5, fr_resized.shape[1] - 0.5]) # there is end effect for calculating ci
ax.set_xticks([0 + 0.5, fr_resized.shape[1] - 0.5])
ax.set_xticklabels([0, 360])
ax.set_ylim([0 + 0.5, fr_resized.shape[0] - 0.5]) # there is end effect for calculating ci
ax.set_yticks([0, (fr_resized.shape[0] - 0.5)])
ax.set_yticklabels(['2', '0'])

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.tick_params(bottom=False, left=False)

ax.set_xlabel('Neurons \n (Labeled by prefered color)')
ax.set_ylabel('Time (seconds)')

cbar = fig.colorbar(im)

#yticks = np.array([0.5, 1.5, 2.5]) * np.exp(1)
#cbar.set_ticks(yticks)
#cbar.set_ticklabels(np.around(np.log(yticks), decimals=1))
#cbar.set_label('Rate (a.u.)', rotation=270, labelpad=15)
cbar.ax.tick_params(direction='out', length=3)

fig.savefig('./figs/fig_collect/bump.pdf', format='pdf')
#plt.show()
