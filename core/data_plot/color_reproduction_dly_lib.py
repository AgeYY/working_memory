from __future__ import division

import torch
import numpy as np
from matplotlib import pyplot as plt


from .. import dataset
from .. import task
from .. import default
from .. import train

import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn.decomposition import PCA
from matplotlib import cm
from core.color_error import Color_error

import os
from scipy import stats
from numba import jit

from .. import run

fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

plt.style.use('default')

def PCA_2d_plot(rule_name, serial_idx=0, prod_intervals=np.array([1200]), epoch='interval', noise_on=False, ring_centers = np.array([6., 12., 18., 24.]), colors=None, is_cuda=False):
    model_dir = '../core/model/'+ rule_name + '/' + str(serial_idx)

    prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)

    runnerObj = run.Runner(model_dir=model_dir, rule_name=rule_name, is_cuda=is_cuda, noise_on=noise_on)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, sampled_degree=ring_centers) # will dly_interval also be passed to the run?

    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
    out_puts = run_result.outputs.detach().cpu().numpy()
    out_puts_argmax = np.argmax(out_puts, axis = 2)


    stim1_off, stim2_on = trial_input.epochs[epoch]
    res_on, res_off = trial_input.epochs['response']

    #if epoch == 'delay':
    #    stim1_off = stim2_on - 1
    stim1_off = stim1_off + 6
    #stim1_off = stim1_off
    # (stimulus, time, neuron)
    firing_rate_list = np.concatenate(list(firing_rate[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :,  :] for i in range(0, batch_size)), axis=0)
    output_list = np.concatenate(list(out_puts_argmax[res_on[i]:res_off[i], i][np.newaxis, :] for i in range(0, batch_size)), axis=0)

    concate_firing_rate = np.reshape(firing_rate_list, (-1, firing_rate_list.shape[-1]))

    pca = PCA(n_components=2)
    pca.fit(concate_firing_rate)
    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    time_size = stim2_on - stim1_off

    delim = np.cumsum(time_size)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:-1], axis=0)

    ##########################################################################################

    #component of time

    concate_firing_rate_time = np.mean(firing_rate_list, axis=0)

    pca_time = PCA(n_components=1)
    pca_time.fit(concate_firing_rate_time)

    ##########################################################################################
    #component of stimulus

    concate_firing_rate_stim = np.mean(firing_rate_list, axis=1)

    pca_stim = PCA(n_components=1)
    pca_stim.fit(concate_firing_rate_stim)

    ##########################################################################################

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax = fig.gca()
    import matplotlib.cm as cm

    for i in range(0, len(concate_transform_split)):
    #for i in range(0, len(ring_centers)):
        if colors is None:
            colori = color_map_color(i/len(concate_transform_split))
        else:
            colori = colors[i, :] # must be RGBA
        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], color=colori, alpha=0.7)
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], marker='*', color=colori, alpha=0.7)
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], marker='o', color=colori, alpha=0.7)

    ax.set_xlabel('PC1', fontsize=fs)
    ax.set_ylabel('PC2', fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    plt.show()
    return fig, output_list

def performence(rule_name, serial_idx=0, prod_intervals=np.array([800]), noise_on=False, ring_centers=np.array([6., 12., 18., 24.]), dire_on=True):
    '''
    dire_on (bool): on -- return output direction and the target direction. off -- return success_action_prob, mean_direction_err
    '''
    model_dir = '../core/model/'+ rule_name + '/' +str(serial_idx)

    prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)

    runnerObj = run.Runner(model_dir=model_dir, rule_name=rule_name, is_cuda=True, noise_on=noise_on)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, gaussian_center=ring_centers, sampled_degree=ring_centers) # will dly_interval also be passed to the run?

    out_puts = run_result.outputs.detach().cpu().numpy()
    out1, out2 = train.get_perf_color(out_puts, rule_name, trial_input.sampled_degree, trial_input.epochs['stim1'][1], trial_input.epochs['go_cue'][1], dire_on=dire_on)

    out1 = out1.tolist()
    out2 = out2.tolist()

    return out1, out2 # if dire_on is ture, output dire. success_action_prob is the 

def show_predition(rule_name, serial_idx=0, batch_size=1, is_cuda=False):
    model_dir = '../core/model/'+ rule_name + '/'+str(serial_idx)
    ring_centers = np.array([30.]); prod_intervals=np.array([800.])
    runnerObj = run.Runner(model_dir=model_dir, rule_name=rule_name, is_cuda=is_cuda, noise_on=False)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, sampled_degree=ring_centers) # will dly_interval also be passed to the run?
    '''features of trial_input refer to dataset --> result['input'] = ...'''
    #print(trial_input.input)
    Y = run_result.outputs.cpu()
    time_len = Y.shape[0]
    X = trial_input.x
    Y_target = trial_input.y
    
    plt.figure(0)
    for unit_id in range(X.shape[2]):
        plt.plot(range(time_len), X[:, 0, unit_id] + np.ones(time_len) * (unit_id - 1))
    plt.title('inputs')
    plt.xlabel('time')
    plt.ylabel('units')

    plt.figure(1)
    for unit_id in range(Y.shape[2]):
        plt.plot(range(time_len), Y[:, 0, unit_id] + np.ones(time_len) * unit_id)
    plt.title('model outputs')
    plt.xlabel('time')
    plt.ylabel('units')

    plt.figure(2)
    for unit_id in range(Y_target.shape[2]):
        plt.plot(range(time_len), Y_target[:, 0, unit_id] + np.ones(time_len) * unit_id)
    plt.title('target outputs')
    plt.xlabel('time')
    plt.ylabel('units')

def bias_around_common(out_color, target_color, common_color):
    '''
    Bias around common color
    Input:
      out_color (np.array [float] [batch_size, n] or [n]): the reported color.
      target_color (np.array [float] [batch_size, n] or [n]): the target color. Note the index must the same with out_color.
      common_color (list [float] [m]): the whole color space (360 deegree) is evenly divided by common colors. Such as [0, 60, 120, 180, 240, 300] or [20, 110, 200, 290]
    Output:
      target_common_color (np.array [float] [batch_size, n] or [n]): The target color but scaled into the interval (-width, +width), where 2 * width = 360 / m.
      bias (np.array [float] [batch_size, n] or [n]): out_color - target_color
    '''
    width = 360 / len(common_color)

    # Handle both 1D and 2D inputs
    out_color = np.array(out_color)
    target_color = np.array(target_color)
    original_shape = out_color.shape
    
    # Reshape to 2D if input is 1D
    if len(out_color.shape) == 1:
        out_color = out_color.reshape(1, -1)
        target_color = target_color.reshape(1, -1)

    # calculate the error
    color_error = Color_error()
    color_error.add_data(out_color.flatten(), target_color.flatten())
    bias = color_error.calculate_error()
    bias = bias.reshape(out_color.shape)

    target_common_color = target_color - common_color[0] + width / 2 # shift the first common color to width / 2
    target_common_color = target_common_color % width - width / 2

    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        target_common_color = target_common_color.flatten()
        bias = bias.flatten()

    return target_common_color, bias

def PCA_3d_plot(model_dir, rule_name, prod_intervals=np.array([1200]), epoch='interval', noise_on=False, ring_centers = np.array([6., 12., 18., 24.]), colors=None, is_cuda=False):

    prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
    prod_intervals = prod_intervals.flatten()
    ring_centers = ring_centers.flatten()

    batch_size = len(prod_intervals)

    runnerObj = run.Runner(model_dir=model_dir, rule_name=rule_name, is_cuda=is_cuda, noise_on=noise_on)
    trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, sampled_degree=ring_centers) # will dly_interval also be passed to the run?

    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
    out_puts = run_result.outputs.detach().cpu().numpy()
    out_puts_argmax = np.argmax(out_puts, axis = 2)


    stim1_off, stim2_on = trial_input.epochs[epoch]
    res_on, res_off = trial_input.epochs['response']

    #if epoch == 'delay':
    #    stim1_off = stim2_on - 1
    stim1_off = stim1_off + 20
    #stim1_off = stim1_off
    # (stimulus, time, neuron)
    firing_rate_list = np.concatenate(list(firing_rate[stim1_off[i]:stim2_on[i], i, :][np.newaxis, :,  :] for i in range(0, batch_size)), axis=0)
    output_list = np.concatenate(list(out_puts_argmax[res_on[i]:res_off[i], i][np.newaxis, :] for i in range(0, batch_size)), axis=0)

    concate_firing_rate = np.reshape(firing_rate_list, (-1, firing_rate_list.shape[-1]))

    pca = PCA(n_components=3)
    pca.fit(concate_firing_rate)
    print('the cumulative variance explained in the first three conponents: ', np.cumsum(pca.explained_variance_ratio_))
    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    time_size = stim2_on - stim1_off

    delim = np.cumsum(time_size)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:-1], axis=0)


    ##########################################################################################
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')

    import matplotlib.cm as cm

    for i in range(0, len(concate_transform_split)):
    #for i in range(0, len(ring_centers)):
        if colors is None:
            colori = color_map_color(i/len(concate_transform_split))
        else:
            colori = colors[i, :] # must be RGBA
        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], concate_transform_split[i][:, 2], color=colori, alpha=0.7)
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], concate_transform_split[i][0, 2], marker='*', color=colori, alpha=0.7)
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], concate_transform_split[i][-1, 2], marker='o', color=colori, alpha=0.7)

    #ax.set_xlabel('PC1', fontsize=fs)
    #ax.set_ylabel('PC2', fontsize=fs)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.grid(False)
    #plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    plt.show()
    return fig, output_list
