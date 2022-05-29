# study the structure property of weight matrix. Neuron are labeled by rnn decoder
import context
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent, Agent_group
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir, sc_dist
import sys
from core.net_struct.main import circular_mean
from core.tools import mean_se
import pandas as pd
import seaborn as sns
import matplotlib
from core.tools import find_nearest
from scipy.stats import vonmises
from core.net_struct.labeling import Labeler

##### input arguments
try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = 'model_16/noise_pos_fir/'

####################
gen_data=True
epoch = 'interval'
batch_size = 1 # batch for searching tuning curve
prod_intervals=200 # set the delay time to 800 ms for ploring the trajectory
n_colors=1000
sigma_rec = None # noise in single trial, not for calculate tunning
sigma_x = None
sigma= 0.1 # noise when do the experiment for searching for tuning curve
bin_width_tuning = 10

tuned_thre = -1 # delete neurons that are inactivated by thrething the tuning curve
bin_width = 10
width_common = 10 # width of common color
common_color = [40, 130, 220, 310]
fs = 10 # fontsize
baseline_left = 0; baseline_right = 0;

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def one_step(sub):
    '''
    input:
    sub (agent)
    return:
    label (array [float]): input color
    cmean_list (list [float]): delta color in the next time step
    '''

    ## get the tunning matrix
    laber = Labeler()
    laber.read_rnn_agent(sub)

    #laber.label_delay(prod_intervals=prod_intervals, n_colors=180)
    laber.label_rnn_decoder(sigma=sigma, batch_size=batch_size, prod_intervals=prod_intervals, n_colors=n_colors, bin_width=bin_width_tuning)

    bump = Bump_activity() # passing parameters to bump so it can do the rest of processing
    bump.tuning = laber.tuning
    bump.label = laber.label
    bump.t_strength = laber.t_strength
    bump.input_colors = laber.sense_color

    weight_hh = sub.model.weight_hh.detach().cpu().numpy()

    weight_pped, label_hh = sc_dist(bump, weight_hh, thre=tuned_thre, bin_width=bin_width)

    print(np.linalg.norm(weight_pped))

    bias_hh = sub.model.bias_h.detach().cpu().numpy()
    bias_hh = bias_hh.reshape(1, -1)
    bias_hh_pped, label_bias = bump_pipline(bump, bias_hh, bin_width=bin_width, thre=tuned_thre)

    weight_mask = np.nan_to_num(weight_pped, 1e10)
    weight_mask = weight_mask < 1e9

    weight_pped_nan = np.nan_to_num(weight_pped, 0)
    bias_hh_pped_nan = np.nan_to_num(bias_hh_pped.reshape(-1), 0)
    weight_cmean = np.eye(weight_pped.shape[0])
    label_good = []
    cmean_err = []
    hidden_size = weight_pped.shape[0]
    for i in range(weight_pped.shape[1]):
        #cmean, norm = circular_mean(np.tanh(weight_pped_nan[i, :] + bias_hh_pped_nan) + 1, label_hh)

        #eq_v = np.tanh(weight_pped_nan[i, :] + bias_hh_pped_nan)

        #sigma = 30
        #kappa = 1 / np.deg2rad(sigma * 2)**2
        #x = label_hh.copy()
        #eq_v_init =  10 * vonmises.pdf(np.deg2rad(x), kappa, loc=np.deg2rad(label_hh[i]))

        eq_v_init = np.zeros(hidden_size)
        eq_v_init[i] = 1 # eq_v_init is firing here

        ########## Test dynamics
        eq_v = np.dot(eq_v_init, weight_pped_nan) + bias_hh_pped_nan # this is state, not firing
        for it in range(0): # iteration on the weight. The target weight is defined as W^n_weight
            eq_v = np.dot(np.tanh(eq_v) + 1, weight_pped_nan) + bias_hh_pped_nan
        #eq_v = np.dot(np.tanh(eq_v_init), weight_pped_nan)
        #for it in range(0): # iteration on the weight. The target weight is defined as W^n_weight
        #    eq_v = np.dot(np.tanh(eq_v), weight_pped_nan)
        ##########

        cmean, norm = circular_mean(
            np.tanh(eq_v) + 1, label_hh
        )

        if norm > 1e-8: # if the norm is too small, we think its due to there's no preferencial neuron within this small color interval
            error = cmean - label_hh[i]
            label_good.append(label_hh[i])
            if error > 180:
                cmean_err.append(error - 360)
            elif error < -180:
                cmean_err.append(error + 360)
            else:
                cmean_err.append(error)

    cmean_err = np.array(cmean_err)

    return label_good, cmean_err, label_hh, weight_pped_nan, weight_mask

if gen_data:
    sub = Agent(model_dir+sub_dir, rule_name)

    label = [] # prefered color in current step
    cmean_arr = [] # prefered color of neurons in the next step
    weight_hh_avg = np.zeros((360 // bin_width, 360 // bin_width)) # the averge recurrent weight
    weight_mask_tt = np.zeros(weight_hh_avg.shape)
    label_temp, cmean_arr_temp, label_hh, weight_pped_nan, weight_mask = one_step(sub)
    weight_mask_tt = weight_mask_tt + weight_mask
    weight_hh_avg = weight_hh_avg + weight_pped_nan

    ########## Test weight
    #eq_weight = np.eye(weight_pped_nan.shape[0])
    #for it in range(100): # iteration on the weight. The target weight is defined as W^n_weight
    #    eq_weight = np.dot(eq_weight, weight_pped_nan)
    #weight_hh_avg = weight_hh_avg + eq_weight
    ##########
    
    label.append(label_temp.copy())
    cmean_arr.append(cmean_arr_temp.copy())

    #weight_hh_avg = weight_hh_avg / weight_mask
    weight_hh_avg = weight_hh_avg / weight_mask_tt
    label = np.concatenate(label, axis=0)
    cmean_arr = np.concatenate(cmean_arr, axis=0)

    pd.DataFrame(label_hh).to_csv('./data/label_hh.csv', header=None, index=None)
    pd.DataFrame(weight_hh_avg).to_csv('./data/weight_hh.csv', header=None, index=None)

    pd.DataFrame(label).to_csv('./data/current_prefer.csv', header=None, index=None)
    pd.DataFrame(cmean_arr).to_csv('./data/next_prefer.csv', header=None, index=None)

##### read data
label_hh = pd.read_csv('./data/label_hh.csv', header=None).to_numpy().reshape(-1)
weight_hh = pd.read_csv('./data/weight_hh.csv', header=None).to_numpy()
label = pd.read_csv('./data/current_prefer.csv', header=None).to_numpy()
cmean_arr = pd.read_csv('./data/next_prefer.csv', header=None).to_numpy()

##### plot
sns.set()
sns.set_style("ticks")

## weight matrix
fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.2, 0.2, 0.65, 0.6])
im = ax.imshow(weight_hh, cmap='bwr')

ax.set_xlim([0 + 0.5, weight_hh.shape[1] - 0.5]) # there is end effect for calculating ci
ax.set_xticks([0 + 0.5, weight_hh.shape[1] - 0.5])
ax.set_xticklabels([0, 360])
ax.set_ylim([0 + 0.5, weight_hh.shape[0] - 0.5]) # there is end effect for calculating ci
ax.set_yticks([0, (weight_hh.shape[0] - 0.5)])
ax.set_yticklabels([0, 360])
ax.tick_params(direction='in')

ax.set_xlabel('Target units \n (Labeled by prefered color)')
ax.set_ylabel('Source units')

cbar = fig.colorbar(im, fraction=0.045)

cbar.set_ticks([-0.015, 0, 0.015])
cbar.set_ticklabels(['-1.5', '0', '1.5'])
cbar.ax.tick_params(direction='out', length=3)

cbar.set_label('Connection strength (a.u.)', rotation=270, labelpad=15)

plt.show()

## change of prefer color
target, mean_error, se_error = mean_se(label.flatten(), cmean_arr.flatten(), remove_outlier=True, m=3.5, sd=False)
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.30, 0.2, 0.6, 0.7])
ax.fill_between(target, mean_error - se_error, mean_error + se_error, alpha=0.4)
ax.plot(target, mean_error)
ax.axhline(y = 0, linestyle = '--', linewidth = 2, color = 'red')

for cc_i in common_color: # indicate four common colors
    idx = np.abs(target - cc_i) < 10
    ax.plot(target[idx], np.zeros(np.sum(idx)), color='black', linewidth = 8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xticks(np.linspace(0, 360, 5))

ax.set_ylabel(r'$c(t+1) - c(t)$' + '\n (or unsymmertric Index)')
ax.set_xlabel('c(t)')

ax.tick_params(direction='in')
# y axis can be interpreted in two ways. The change of color after one step, or unsymmertric index of the output weight of unit who is labeled by c(t)

plt.show()

deg_len = len(weight_hh[0])
weight_roll = np.empty(weight_hh.shape)
for i, w_i in enumerate(weight_hh):
    w_i_roll = np.roll(w_i, deg_len // 2 - i)
    weight_roll[i] = np.roll(w_i, deg_len // 2 - i)

def plot_sc(weight_roll):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.25, 0.2, 0.6, 0.6])

    weight_std = np.std(weight_roll, axis=0)
    weight_mean = np.mean(weight_roll, axis=0)

    ax.fill_between(label_hh - 180, weight_mean - weight_std, weight_mean + weight_std, alpha = 0.3)
    ax.plot(label_hh - 180, weight_mean)
    ax.axvline(x = 0, linestyle = '--', linewidth = 3, color = '#E50000')
    ax.axhline(y = 0, linestyle = '--', linewidth = 3, color = 'grey')
    ax.set_xlabel('Difference of Preferential Angle')
    ax.set_ylabel('Connection Strength (a.u.)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in')

    return fig, ax

##### averge connection
fig, ax = plot_sc(weight_roll)
plt.show()

##### bias connection left
def bias_sc(dire='left', baseline=0):
    '''
    output the weight matrix of subset of neurons. If dire=='left', the neurons are from [- (360 / common_color / 2) + baseline, -width_common - baseline], where 0 is the common color. If dire == 'right', the range is [width_common + baseline, (360 / len(common_color) / 2) - baseline].

    input:
    baseline (float): see the above. This is used for avoid end effect
    '''
    if dire == 'left':
        left_bound = - (360 / len(common_color) / 2)
        deg_ran = [left_bound + baseline, -width_common - baseline]
        def dis_color(label, common):
            return (label - common) % 360 - 360
    elif dire == 'right':
        right_bound = (360 / len(common_color) / 2)
        deg_ran = [width_common + baseline, right_bound - baseline]
        def dis_color(label, common):
            return (label - common) % 360

    index = np.full(weight_roll.shape[0], False, dtype=bool)
    for cc in common_color:
        #dis = (label_hh - cc) % 360 - 360 # circular minus. All results are negtive
        dis = dis_color(label_hh, cc)
        index = index + (dis > deg_ran[0]) * ( dis < deg_ran[1])
    weight_shift = weight_roll[index]
    return weight_shift

weight_left = bias_sc('left', baseline=baseline_left)
fig, ax = plot_sc(weight_left)
plt.show()

weight_right = bias_sc('right', baseline=baseline_right)
fig, ax = plot_sc(weight_right)
plt.show()
