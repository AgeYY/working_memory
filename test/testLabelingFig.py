# Figures testing labeling.py
import context
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent, Agent_group
from core.net_struct.labeling import Labeler, array_pipline
from core.tools import find_nearest
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir, sc_dist
from core.tools import mean_se
import seaborn as sns


sns.set()
sns.set_style('ticks')

model_dir = '../core/model_local/color_reproduction_delay_unit/'
#sub_dir = 'model_16/noise_delta_stronger/'
sub_dir = 'model_16/noise_new/'
rule_name = 'color_reproduction_delay_unit'
prod_intervals=20 # set the delay time to 800 ms for ploring the trajectory
n_colors=180
tuned_thre = 0.5

agent_path = model_dir + sub_dir
sub = Agent(agent_path, rule_name)
laber = Labeler()
laber.read_rnn_agent(sub)

color_min, color_max = 180, 185

state_list = laber.do_exp(sigma=0.05, batch_size=1, prod_intervals=1000, n_colors=360) # generate data
report_color = laber.decode_data(state_list) # decode data to color
tuning, sense_color = laber.compute_tuning(state_list, report_color, bin_width=2) # neural response to different colors, where colors are bined
label, t_strength = laber.prefer_color(tuning, sense_color) # labeling neurons with prefer colors

###### Tuning curve for example neuron
#fir = np.tanh(state_list)
#bin_width = 5
#report_color_bin = report_color // bin_width * bin_width
#target, mean, se_error = mean_se(report_color_bin, fir[:, idx_prefer]) # ith neural tuning
#fig = plt.figure(figsize=(3, 3))
#ax = fig.add_axes([0.30, 0.2, 0.6, 0.7])
#ax.fill_between(target, mean - se_error, mean + se_error, alpha=0.4)
#ax.plot(target, mean)
#ax.set_xlabel('Encoding Color')
#ax.set_ylabel('Neural Firing Rate')
#plt.show()
#
###### Tuning curves of all neurons
#tuning_sort, label_sort = array_pipline(laber.tuning, laber.label, thresh=-999, bin_width=None)
#
#fig = plt.figure(figsize = (3, 3))
#ax = fig.add_subplot(111)
#ax.imshow(tuning_sort, cmap='viridis')
#plt.show()

sub = laber.sub
target_color = sub.behaviour['target_color']
target_color_idx = (target_color < color_max) * (target_color > color_min) # example trials whose input color within the interval
target_color_idx_exp = np.argmax(target_color_idx) # only consider one example color

def example_fir(prefer_color):
    '''
    example neuron's firing whose prefer color is prefer_color)
    input:
      prefer_color (float)
    output:
      exp_fir (array [time_len]): firing of that example neuron
    '''
    idx_prefer = find_nearest(label, prefer_color) # neural prefer that target color

    exp_fir = sub.fir_rate[:, target_color_idx_exp, idx_prefer] # example firing rate for one example neuron
    return exp_fir

n_neuron = 4
exp_fir_list = [[]] * n_neuron
exp_fir_list[0] = example_fir((color_max + color_min) / 2)
exp_fir_list[1] = example_fir(140)
exp_fir_list[2] = example_fir(200)
exp_fir_list[3] = example_fir(260)

time = np.arange(exp_fir_list[0].shape[0])

fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.2, 0.2, 0.65, 0.6])

for efir in exp_fir_list:
    ax.plot(time * 20, efir + 1) # +1 is to shift baseline to avoid negtive firing.

time_point = np.zeros(4, dtype=np.int) # four important time points: perception start, perception_end, delay end, go cue end
time_point[0] = sub.epochs['fix'][1]
time_point[1] = sub.epochs['stim1'][1]
time_point[2] = sub.epochs['interval'][1]
time_point[3] = sub.epochs['go'][1]

for tp in time_point:
    ax.axvline(x = tp * 20, linestyle = '--', linewidth = 3, color = '#E50000')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate')
plt.show()


########## Firing distribution
## print the distribution of firing for neuron 1 in response to color 5 - 10
response_idx = (report_color < color_max) * (report_color > color_min) # find the index in response to colors between color_min and color_max

# find neuron prefer to color 260
target_neuron = 260
idx_prefer = find_nearest(label, target_neuron)

states = state_list[response_idx, idx_prefer]
efir_delay_rnn = np.tanh(states)

efir = example_fir(target_neuron)
efir_delay = efir[time_point[1]:time_point[2]] # firing in the delay

fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.2, 0.2, 0.65, 0.6])

sns.histplot(data= efir_delay + 1, stat='density', ax=ax, color='blue', alpha=0.4)
sns.histplot(data= efir_delay_rnn + 1, stat='density', ax=ax, color='orange', alpha=0.4)
plt.xlabel('firing rate')
plt.ylabel('density')
plt.show()
