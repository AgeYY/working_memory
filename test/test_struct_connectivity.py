# This file calculate the structure connectivity of an RNN.
# 1. Do experiments with different input colors
# 2. Collect firing rates, label neurons with their preferred color
# 3. Obtain weight matrix of the RNN
# 4. If bin_width is not None, the result weight_pped is the averaged structure connectivity. It is a matrix of shape (n_color, n_color), each element is the connection strength from neuron preferred to color 1 to neurons preferred to color 2. For example, the RNN may have 10 neurons prefer color 100, 3 neurons prefer color 5, then the connectivity from color 100 to color 5 is the averge of 3 * 100 connections.
import os
import context
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from core.agent import Agent
from core.net_struct.struct_analyzer import Struct_analyzer

model_dir_parent = '../core/model/model_3.0/color_reproduction_delay_unit/' # one rnn model
model_dir = model_dir_parent + 'model_0/'
rule_name = 'color_reproduction_delay_unit' # rule name is fixed to color_reproduction_delay_unit. Actually this repo can also train another type of RNN with slightly different input format, but in this paper we only use color_reproduction_delay_unit
prod_intervals=800 # set the delay time to 800 ms for ploring the trajectory
n_colors=1000 # number of trials, each trial with different color
#tuned_thre = -0.0 # discard neurons with weak color selectivity, for example, a flat tuning curve
tuned_thre = 0.5 # discard neurons with weak color selectivity, for example, a flat tuning curve

bin_width_color = 1 # We mesh color bins in calculating tuning curve.
bin_width_color_avg_method = 'gaussian'

#bin_width = None # connectivity of these neurons with similar preferred color would be averaged. Note bin_width_color is mesh for tuning curve, but this one is mesh for preferred color.
bin_width = None # connectivity of these neurons with similar preferred color would be averaged. Note bin_width_color is mesh for tuning curve, but this one is mesh for preferred color.
method = 'rnn_decoder' # use mean or max of firing rate to label neuron
label_method = 'max' # use mean or max of firing rate to label neuron
nan_method = 'remove' # how to handle nan ==> remove it
generate_state_method = 'delay_ring'
num_rnn_max = 5

# doing some experiments to collect firing rates

def compute_one_sub_connect(sub):
    str_ana = Struct_analyzer()
    str_ana.read_rnn_agent(sub)
    _, _, _, t_strength = str_ana.prepare_label(n_colors=n_colors, sigma_rec=0, sigma_x=0, batch_size=1, prod_intervals=prod_intervals, method=method, bin_width_color=bin_width_color, nan_method=nan_method, generate_state_method=generate_state_method, label_method=label_method, bin_width_color_avg_method=bin_width_color_avg_method)
    #plt.figure()
    #plt.hist(t_strength)
    #plt.show()
    weight_hh_pped, label_weight_pped = str_ana.output_weight(thresh=tuned_thre, bin_width=bin_width)
    bias_hh_pped, label_bias_pped = str_ana.output_bias(thresh=tuned_thre, bin_width=bin_width)
    return weight_hh_pped, label_weight_pped, bias_hh_pped, label_bias_pped


weight_hh_all, label_weight_all = [], []
bias_hh_all = []
count = 0
for filename in os.listdir(model_dir_parent):
    f = os.path.join(model_dir_parent, filename)
    sub = Agent(f, rule_name)
    weight_hh_pped, label_weight_pped, bias_hh_pped, label_bias_pped = compute_one_sub_connect(sub)
    weight_hh_all.append(weight_hh_pped)
    label_weight_all.append(label_weight_pped)
    bias_hh_all.append(bias_hh_pped)
    count += 1
    if count > num_rnn_max: break

#weight_hh_all = [np.mean(weight_hh_all, axis=0)] # replace with avg
#label_weight_all = [np.mean(label_weight_all, axis=0)] # replace with avg

## draw weight matrix
from matplotlib import colors
divnorm=colors.TwoSlopeNorm(vcenter=0.)
#plt.figure()
#plt.imshow(np.mean(weight_hh_all, axis=0), cmap='seismic', norm=divnorm)
#plt.show()
#
#plt.figure()
#for label in label_weight_all:
#    plt.scatter(np.arange(256), label)
#plt.show()
for i, weight in enumerate(weight_hh_all):
    plt.figure()
    plt.imshow(weight, cmap='seismic', norm=divnorm)

    plt.figure()
    plt.scatter(np.arange(len(label_weight_all[i])), label_weight_all[i])
    plt.show()

## plot label distribution
#plt.figure()
#label_weight_all_flat = np.array(label_weight_all).flatten()
#plt.hist(label_weight_all_flat, bins=100)
#plt.show()

# compute excitation
num_bin = 360 // bin_width_color

color_bin = np.linspace(0, 360, num_bin + 1, endpoint=True) # +1 to include endpoint
color_bin_label = color_bin + bin_width_color / 2.0 # use center to label each bin
color_bin_label = color_bin_label[:-1] # drop out the last label

group_rnn_counter = np.zeros(num_bin)
sum_group_ext = np.zeros(num_bin)

for i, weight_hh_i in enumerate(weight_hh_all):
    num_neuron = len( label_weight_all[i] )
    idx = np.digitize(label_weight_all[i], color_bin) - 1 # map labels to bin index

    #weight_hh_i[weight_hh_i<0] = 0
    sum_ext_i_index_by_label = np.sum(weight_hh_i, axis=0)

    sum_ext_i = np.zeros(num_bin)
    single_rnn_counter = np.zeros(num_bin)
    for i in range(num_neuron):
        sum_ext_i[idx[i]] += sum_ext_i_index_by_label[i]
        single_rnn_counter[idx[i]] += 1

    avg_ext_i = np.divide(sum_ext_i, single_rnn_counter, out=np.zeros_like(sum_ext_i), where=single_rnn_counter!=0)

    sum_group_ext += avg_ext_i
    group_rnn_counter += (single_rnn_counter >= 1)

avg_group_ext = np.divide(sum_group_ext, group_rnn_counter, out=np.zeros_like(sum_group_ext), where=group_rnn_counter!=0)
from scipy.ndimage import gaussian_filter1d
avg_group_ext = gaussian_filter1d(avg_group_ext, 3, mode='wrap')

plt.figure()
plt.scatter(color_bin_label, avg_group_ext)
plt.show()

#common_color = np.array([40, 130, 220, 310]) / bin_width + 0.5
#weight_input = np.sum(weight_hh_avg, axis=1)
#plt.figure()
#plt.scatter(range(weight_input.shape[0]), weight_input)
#for cc in common_color:
#    plt.axvline(cc)
#plt.show()
