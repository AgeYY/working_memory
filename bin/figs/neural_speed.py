# neural speed during the delay
import context
import pandas as pd
import sys
from core.agent import Agent, Agent_group
import numpy as np
import matplotlib.pyplot as plt
import core.tools as tools
import seaborn as sns
from sklearn.decomposition import PCA

try:
    rule_name = sys.argv[1]
    sub_dir = sys.argv[2]
    out_path = sys.argv[4]
except:
    rule_name = 'color_reproduction_delay_unit'
    out_path = './figs/fig_data/neural_speed.pdf'
    sub_dir = '/noise_delta'

try:
    if sys.argv[3] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

batch_size = 100
max_sub = 50
prod_intervals = 1000
t_cutoff = 200 # time cutoff for stimulus and overall plot
estimator=np.mean
# keys are used to plot speed-distribution
keys = ['90.0', '30.0', '27.5', '25.0', '22.5', '20.0', '17.5', '15.0', '12.5', '10.0', '3.0']
keys_num = keys.copy()
# keys part are used for ploting speed-stimulus and speed-time
keys_part = ['90.0', '25.0', '10.0', '3.0']
keys_part_label = [r'$90^{\circ}$', r'$25^{\circ}$', r'$20^{\circ}$', r'$10^{\circ}$', r'$3^{\circ}$']
ring_centers = np.linspace(0, 360, batch_size)
#ring_centers = np.concatenate((ring_centers, np.array([85, 175, 265, 355])), axis=None) # add four saddle points
#batch_size = batch_size + 4
ci=95
def one_sub_speed(sub, cutoff=100, dt=20):
    '''
    calculate one subjects neural state speed from cutoff to the end of the delay
    output:
      speed_sub (array [float] (n_trials, n_times)): speed of a subject of different trials
    '''

    dly_time = sub.epochs['interval']
    fir_rate_dly = sub.state[dly_time[0]:dly_time[1], :, :]

    speed_sub = []
    for trial in range(fir_rate_dly.shape[1]):
        vel = np.diff(fir_rate_dly[cutoff//dt:, trial, :], axis=0) / 20
        speed = np.linalg.norm(vel, axis=1)
        speed_sub.append(speed.copy())
    speed_sub = np.stack(speed_sub)
    return speed_sub

def group_speed(group, max_sub=1000, dly_cutoff=0, dt=20, ring_centers=[]):
    '''
    output:
      speed_group (array [float] (n_sub, n_trials, n_times)): speed of the group
    '''
    speed_group = []
    count=1
    for sub in group.group:
        sub.do_exp(prod_intervals, ring_centers, sigma_rec=0, sigma_x=0)
        speed_sub = one_sub_speed(sub, cutoff=dly_cutoff)
        speed_group.append(speed_sub.copy())

        count = count + 1
        if count > max_sub:
            break

    speed_group = np.stack(speed_group)
    return speed_group

#################### compute the data
if gen_data:
    data_dis_origin = {} # 90, 25, 20, 3
    for ky in keys:
        model_dir = "../core/model/model_" + ky + "/color_reproduction_delay_unit/"
        group = Agent_group(model_dir, rule_name, sub_dir)
        data_dis_origin[ky] = group_speed(group, max_sub=max_sub, ring_centers=ring_centers)
    tools.save_dic(data_dis_origin, fname=out_path)

data_dis_origin = tools.load_dic(out_path)
for key, value in data_dis_origin.items():
    data_dis_origin[key] = np.array(value, dtype=float)

########## different stimulus but taking into accout of different time and RNNs
data_dis = data_dis_origin.copy()
for key, value in data_dis.items():
    value = value[:, :, t_cutoff//20:]
    value = np.moveaxis(value, 1, 0) # move the axis so that (n_sub, n_trial, n_time) becomes (n_sub, n_time, n_trial)
    value = np.reshape(value, (batch_size, -1))
    value_repeat = value.shape[1]
    data_dis[key] = value.flatten()

ring_centers_mat = ring_centers.reshape(-1, 1)
ring_centers_mat = np.tile(ring_centers_mat, (1, value_repeat))
data_dis['trial'] = ring_centers_mat.flatten()

cdf = pd.DataFrame(data_dis)
mdf = pd.melt(cdf, value_vars=keys_part, var_name='speed_of_distribution', value_name='Speed (deg/ms)')
mdf['Stimulus (deg)'] = np.tile(data_dis['trial'], len(keys_part))

sns.set_style("ticks")
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.2, 0.3, 0.63, 0.6])
sns.lineplot(x='Stimulus (deg)', y='Speed (deg/ms)', hue='speed_of_distribution', data=mdf, ax=ax, ci=ci, estimator=estimator)
#ax.legend(labels=keys_part_label)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='in')
ax.set_xticks(np.linspace(0, 360, 7))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

fig.savefig('./figs/fig_collect/speed_dis_stimuli.pdf', format='pdf')
########## different time but taking into accout of different stimuli and RNNs
data_dis = data_dis_origin.copy()
n_time = data_dis['90.0'].shape[2] - 150//20
start_time = prod_intervals - 20 - (n_time) * 20 # avoid end effect. dt = 20ms
time_series = np.linspace(150, prod_intervals - 20, n_time)

for key, value in data_dis.items():
    value = value[:, :, 150//20:]
    value = np.moveaxis(value, -1, 0) # move the axis so that (n_sub, n_trial, n_time) becomes (n_sub, n_time, n_trial)
    value = np.reshape(value, (n_time, -1))
    value_repeat = value.shape[1]
    data_dis[key] = value.flatten()

time_series_mat = time_series.reshape(-1, 1)
time_series_mat = np.tile(time_series_mat, (1, value_repeat))
data_dis['time'] = time_series_mat.flatten()

cdf = pd.DataFrame(data_dis)
mdf = pd.melt(cdf, value_vars=keys_part, var_name='speed_of_distribution', value_name='Speed (deg/ms)')
mdf['Delay time (ms)'] = np.tile(data_dis['time'], len(keys_part))

sns.set_style("ticks")
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.2, 0.3, 0.63, 0.6])
sns.lineplot(x='Delay time (ms)', y='Speed (deg/ms)', hue='speed_of_distribution', data=mdf, ax=ax, ci=ci, estimator=estimator)
#ax.legend(labels=keys_part_label)
ax.set_xlabel('Delay time (ms)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='in')
ax.axhline(y=0, linestyle = '--', color = 'black')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

fig.savefig('./figs/fig_collect/speed_dis_time.pdf', format='pdf')

########## all
data_dis = data_dis_origin.copy()

for key, value in data_dis.items():
    value = value[:, :, t_cutoff//20:]
    data_dis[key] = value.flatten()

    #value = value[:, :, t_cutoff//20:]
    #value = np.moveaxis(value, 1, 0) # move the axis so that (n_sub, n_trial, n_time) becomes (n_trial, n_sub, n_time)
    #value = np.reshape(value, (batch_size, -1))
    #data_dis[key] = value[0, :]

cdf = pd.DataFrame(data_dis)
mdf = pd.melt(cdf, value_vars=keys, var_name='Distribution', value_name='Speed (deg/ms)')

# replacing the distribution name with numerical entropy/sigma
#for i, ky in enumerate(keys):
#    mdf.replace(ky, keys_num[i], inplace=True)

mdf['Distribution'] = np.array(mdf['Distribution'], dtype=float)

sns.set_style("ticks")
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(3,3))
#sns.boxplot(x='Distribution', y='Speed (deg/ms)', data=mdf, ax=ax, showfliers = False)
sns.lineplot(x='Distribution', y='Speed (deg/ms)', data=mdf, ax=ax1, err_style='bars', ci=ci, marker="o", estimator=estimator)

sns.lineplot(x='Distribution', y='Speed (deg/ms)', data=mdf, ax=ax2, err_style='bars', ci=ci, marker="o", estimator=estimator)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(direction='in')

ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(direction='in', left=False)

ax1.set_xlim(0, 32)
ax1.set_xticks([0, 8, 16, 22, 30])
ax1.set_xlabel(r'$\sigma_s$ of the prior distribution')
ax2.set_xlim(88, 92)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
fig.savefig('./figs/fig_collect/speed_dis_all.pdf', format='pdf')

# same as above, but zoom in to show the error bar
fig_inset = plt.figure(figsize=(3, 3))
ax_inset = fig_inset.add_axes([0.2, 0.3, 0.63, 0.6])
sns.lineplot(x='Distribution', y='Speed (deg/ms)', data=mdf, ax=ax_inset, err_style='bars', ci=ci, marker="o", estimator=estimator)
ax_inset.set_xlim(24.5, 25.5)
ax_inset.set_ylim(0.00379, 0.00384)
ax_inset.tick_params(direction='in')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

fig_inset.savefig('./figs/fig_collect/speed_dis_all_inset.pdf', format='pdf')

#plt.show()
